"""Linear issue lookup for PRs that reference a tracked ticket.

Mira extracts issue identifiers (e.g. ``ENG-123``) from the PR title,
description, and head branch, then fetches the matching Linear issue so the
reviewer can check the change against the ticket's intent and surface the
link in the walkthrough. Everything here is best-effort: a missing key, an
unknown identifier, or any network failure returns no issues and never blocks
a review.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from mira.models import LinkedIssue, PRInfo

if TYPE_CHECKING:
    from mira.config import MiraConfig

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0
_MAX_DESCRIPTION_CHARS = 2_000

# Linear identifiers are an uppercase team key, a dash, and a number. Common
# technical tokens (UTF-8, ISO-8601, CVE-2024-1234, …) match the same shape, so
# they are filtered out rather than sent to the API as doomed lookups.
_ISSUE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})-(\d+)\b")
_IGNORED_PREFIXES = frozenset(
    {
        "UTF",
        "ISO",
        "RFC",
        "CVE",
        "GHSA",
        "SHA",
        "TLS",
        "HTTP",
        "HTTPS",
        "IPV",
        "AES",
        "RSA",
        "MD",
        "PR",
        "SRC",
        "URL",
        "API",
        "SDK",
    }
)

_QUERY = """
query MiraIssue($id: String!) {
  issue(id: $id) {
    identifier
    title
    url
    description
    priorityLabel
    state { name type }
    assignee { displayName }
    labels { nodes { name } }
    comments(first: 30) {
      nodes { body user { displayName } }
    }
    children(first: 30) {
      nodes { identifier title state { name } }
    }
  }
}
"""

# A markdown task-list item: "- [ ] do the thing" / "- [x] done".
_CHECKBOX_RE = re.compile(r"^\s*[-*+]\s*\[[ xX]\]\s+(.+?)\s*$", re.MULTILINE)
# A top-level heading and a bullet/numbered item, for derived criteria sections.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_CRITERIA_HEADING_KEYWORDS = (
    "acceptance criteria",
    "acceptance criteria:",
    "requirements",
    "definition of done",
    "success criteria",
    "acceptance tests",
    "dod",
    "ac:",
)
_MAX_CRITERIA = 25
_MAX_CRITERIA_CHARS = 500


def extract_acceptance_criteria(description: str) -> list[str]:
    """Pull explicit requirements out of a ticket description.

    Prefers markdown task-list items (``- [ ]`` / ``- [x]``), which is how
    Linear tickets usually encode acceptance criteria. When there are none,
    falls back to the bullets under a heading like "Acceptance Criteria" /
    "Requirements" / "Definition of Done". Returns ``[]`` when the ticket has
    no explicit criteria — the verification pass then derives them from the
    description instead.
    """
    if not description:
        return []

    checkboxes = [m.group(1) for m in _CHECKBOX_RE.finditer(description) if m.group(1).strip()]
    if checkboxes:
        return [_clip(c) for c in checkboxes[:_MAX_CRITERIA]]

    lines = description.splitlines()
    for i, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if not heading:
            continue
        title = heading.group(1).strip().lower()
        if not any(kw in title for kw in _CRITERIA_HEADING_KEYWORDS):
            continue
        criteria: list[str] = []
        for raw in lines[i + 1 :]:
            if _HEADING_RE.match(raw):
                break
            # Skip checkbox items here — handled above when present at all.
            if _CHECKBOX_RE.match(raw):
                continue
            bullet = _BULLET_RE.match(raw)
            if bullet and bullet.group(1).strip():
                criteria.append(bullet.group(1).strip())
            if len(criteria) >= _MAX_CRITERIA:
                break
        if criteria:
            return [_clip(c) for c in criteria]
    return []


def _clip(text: str, limit: int = _MAX_CRITERIA_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def extract_issue_identifiers(text: str, team_keys: list[str] | None = None) -> list[str]:
    """Return unique, denylisted Linear identifiers found in ``text``.

    When ``team_keys`` is non-empty only identifiers for those teams are
    returned — the operator declares which team keys are real, which avoids
    probing the API for every ``ABC-123``-shaped token in a description.
    """
    if not text:
        return []
    allowed = {key.upper() for key in team_keys} if team_keys else None
    found: list[str] = []
    for match in _ISSUE_RE.finditer(text):
        key = match.group(1)
        if key in _IGNORED_PREFIXES:
            continue
        if allowed is not None and key not in allowed:
            continue
        identifier = f"{key}-{match.group(2)}"
        if identifier not in found:
            found.append(identifier)
    return found


def issue_identifiers_for_pr(pr_info: PRInfo, team_keys: list[str] | None = None) -> list[str]:
    """Extract issue identifiers from a PR's title, description, and branch."""
    haystack = "\n".join([pr_info.title, pr_info.description, pr_info.head_branch])
    return extract_issue_identifiers(haystack, team_keys)


def _truncate(text: str, limit: int = _MAX_DESCRIPTION_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class LinearClient:
    """Minimal Linear GraphQL client for issue lookups."""

    def __init__(self, api_key: str, api_url: str = "https://api.linear.app/graphql") -> None:
        self._api_key = api_key
        self._api_url = api_url

    async def fetch_issues(self, identifiers: list[str]) -> tuple[list[LinkedIssue], list[str]]:
        """Fetch each identifier.

        Returns the issues Linear knows plus a description of every identifier
        that could not be read, so a caller can tell "no such ticket" apart from
        "the lookup itself failed" instead of reporting a bare empty result.
        """
        issues: list[LinkedIssue] = []
        errors: list[str] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            for identifier in identifiers:
                issue, error = await self._fetch_one(client, identifier)
                if issue is not None:
                    issues.append(issue)
                elif error:
                    errors.append(error)
        return issues, errors

    async def _fetch_one(
        self, client: httpx.AsyncClient, identifier: str
    ) -> tuple[LinkedIssue | None, str]:
        try:
            response = await client.post(
                self._api_url,
                json={"query": _QUERY, "variables": {"id": identifier}},
                headers={
                    # Linear personal API keys are sent raw (no Bearer prefix).
                    "Authorization": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.warning("Linear lookup failed for %s: HTTP %s", identifier, status)
            return None, f"{identifier}: Linear API returned HTTP {status}"
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Linear lookup failed for %s: %s", identifier, exc)
            return None, f"{identifier}: Linear API request failed"

        graphql_errors = payload.get("errors") if isinstance(payload, dict) else None
        if isinstance(graphql_errors, list) and graphql_errors:
            first = graphql_errors[0]
            message = str(first.get("message", "")) if isinstance(first, dict) else str(first)
            logger.warning("Linear lookup rejected for %s: %s", identifier, message)
            return None, f"{identifier}: {message or 'Linear API error'}"

        data = payload.get("data") if isinstance(payload, dict) else None
        issue = data.get("issue") if isinstance(data, dict) else None
        if not isinstance(issue, dict):
            return None, f"{identifier} not found (or not visible to this API key)"
        state = issue.get("state")
        assignee = issue.get("assignee")
        description = str(issue.get("description") or "")

        labels: list[str] = []
        label_nodes = (
            (issue.get("labels") or {}).get("nodes")
            if isinstance(issue.get("labels"), dict)
            else None
        )
        if isinstance(label_nodes, list):
            labels = [
                str(n.get("name")) for n in label_nodes if isinstance(n, dict) and n.get("name")
            ]

        comments: list[str] = []
        comment_nodes = (
            (issue.get("comments") or {}).get("nodes")
            if isinstance(issue.get("comments"), dict)
            else None
        )
        if isinstance(comment_nodes, list):
            for node in comment_nodes:
                if not isinstance(node, dict):
                    continue
                body = _truncate(str(node.get("body") or ""), 600)
                if not body:
                    continue
                user = node.get("user")
                who = str(user.get("displayName") or "") if isinstance(user, dict) else ""
                comments.append(f"{who}: {body}" if who else body)

        children: list[str] = []
        child_nodes = (
            (issue.get("children") or {}).get("nodes")
            if isinstance(issue.get("children"), dict)
            else None
        )
        if isinstance(child_nodes, list):
            for node in child_nodes:
                if not isinstance(node, dict):
                    continue
                child_state = node.get("state")
                state_name = (
                    str(child_state.get("name") or "") if isinstance(child_state, dict) else ""
                )
                summary = f"{node.get('identifier', '')}: {node.get('title', '')}".strip(": ")
                if state_name:
                    summary += f" ({state_name})"
                if summary.strip():
                    children.append(summary)

        return LinkedIssue(
            identifier=str(issue.get("identifier") or identifier),
            title=str(issue.get("title") or ""),
            url=str(issue.get("url") or ""),
            state=str(state.get("name") or "") if isinstance(state, dict) else "",
            state_type=str(state.get("type") or "") if isinstance(state, dict) else "",
            description=_truncate(description),
            criteria=extract_acceptance_criteria(description),
            labels=labels,
            priority=str(issue.get("priorityLabel") or ""),
            assignee=str(assignee.get("displayName") or "") if isinstance(assignee, dict) else "",
            comments=comments,
            children=children,
        ), ""


@dataclass
class LinearLookup:
    """Outcome of a best-effort Linear lookup for a PR's linked tickets."""

    issues: list[LinkedIssue] = field(default_factory=list)
    # "loaded" (issues fetched) | "unavailable" (referenced, but unreadable) |
    # "not_linked" (the PR references no ticket at all).
    status: str = "not_linked"
    # Identifiers the PR references, whether or not Linear could return them.
    identifiers: list[str] = field(default_factory=list)
    # Operator-actionable reason an unavailable lookup failed.
    detail: str = ""


def api_key_env_names(config: MiraConfig) -> list[str]:
    """Env vars checked for the Linear API key, primary first."""
    linear = config.linear
    names = [str(linear.api_key_env)]
    names.extend(str(name) for name in linear.api_key_env_fallbacks)
    return [name for name in names if name]


def _api_key(config: MiraConfig) -> str:
    for name in api_key_env_names(config):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


async def resolve_linked_issues(pr_info: PRInfo, config: MiraConfig) -> LinearLookup:
    """Best-effort fetch of Linear issues referenced by ``pr_info``.

    Failures are reported, not hidden: an empty result carries the reason (no
    API key, disabled, API error, ticket not found/visible) so the review can
    say *why* a referenced ticket is unverified. A tracker lookup must never
    block or break a review, so no failure raises.
    """
    try:
        linear = config.linear
        identifiers = issue_identifiers_for_pr(pr_info, linear.team_keys)
        if not identifiers:
            return LinearLookup(status="not_linked")
        if not linear.enabled:
            return LinearLookup(
                status="unavailable",
                identifiers=identifiers,
                detail="Linear linking is disabled (linear.enabled=false)",
            )
        api_key = _api_key(config)
        if not api_key:
            expected = " or ".join(f"`{name}`" for name in api_key_env_names(config))
            return LinearLookup(
                status="unavailable",
                identifiers=identifiers,
                detail=f"no Linear API key set (expected {expected})",
            )
        issues, errors = await LinearClient(api_key, str(linear.api_url)).fetch_issues(identifiers)
        if issues:
            return LinearLookup(status="loaded", identifiers=identifiers, issues=issues)
        return LinearLookup(
            status="unavailable",
            identifiers=identifiers,
            detail="; ".join(errors) or "Linear returned no matching issue",
        )
    except Exception as exc:  # noqa: BLE001 — never let ticket lookup break a review
        logger.warning("Linear issue resolution failed: %s", exc)
        return LinearLookup(
            status="unavailable",
            detail=f"Linear lookup failed: {exc}",
        )


def format_issues_context(issues: list[LinkedIssue]) -> str:
    """Render linked issues as prompt context for the reviewer."""
    if not issues:
        return ""
    lines = [
        "## Linked Issues",
        "",
        "This PR references the tracker issues below. Check the diff against every "
        "stated requirement and acceptance criterion. Flag an inline comment when "
        "the change contradicts the ticket, leaves a stated requirement unmet, or "
        "implements something the ticket did not ask for. Do not restate the "
        "ticket back to the author.",
        "",
    ]
    for issue in issues:
        header = f"### {issue.identifier}"
        if issue.title:
            header += f" — {issue.title}"
        lines.append(header)
        meta: list[str] = []
        if issue.state:
            meta.append(f"**State:** {issue.state}")
        if issue.priority:
            meta.append(f"**Priority:** {issue.priority}")
        if issue.assignee:
            meta.append(f"**Assignee:** {issue.assignee}")
        if issue.labels:
            meta.append(f"**Labels:** {', '.join(issue.labels)}")
        if meta:
            lines.append("- " + " · ".join(meta))
        if issue.url:
            lines.append(f"- **URL:** {issue.url}")
        if issue.description:
            lines.append("")
            lines.append(issue.description)
        if issue.criteria:
            lines.append("")
            lines.append("**Acceptance criteria (each must be satisfied):**")
            lines.extend(f"- {c}" for c in issue.criteria)
        if issue.children:
            lines.append("")
            lines.append("**Sub-issues:**")
            lines.extend(f"- {c}" for c in issue.children)
        if issue.comments:
            lines.append("")
            lines.append("**Recent ticket comments:**")
            lines.extend(f"- {c}" for c in issue.comments)
        lines.append("")
    return "\n".join(lines).rstrip()
