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
    state { name }
  }
}
"""


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

    async def fetch_issues(self, identifiers: list[str]) -> list[LinkedIssue]:
        """Fetch each identifier, skipping ones Linear doesn't know."""
        issues: list[LinkedIssue] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            for identifier in identifiers:
                issue = await self._fetch_one(client, identifier)
                if issue is not None:
                    issues.append(issue)
        return issues

    async def _fetch_one(self, client: httpx.AsyncClient, identifier: str) -> LinkedIssue | None:
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
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Linear lookup failed for %s: %s", identifier, exc)
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        issue = data.get("issue") if isinstance(data, dict) else None
        if not isinstance(issue, dict):
            return None
        state = issue.get("state")
        return LinkedIssue(
            identifier=str(issue.get("identifier") or identifier),
            title=str(issue.get("title") or ""),
            url=str(issue.get("url") or ""),
            state=str(state.get("name") or "") if isinstance(state, dict) else "",
            description=_truncate(str(issue.get("description") or "")),
        )


async def resolve_linked_issues(pr_info: PRInfo, config: MiraConfig) -> list[LinkedIssue]:
    """Best-effort fetch of Linear issues referenced by ``pr_info``.

    Every failure mode (disabled, missing key, bad config, network) resolves to
    an empty list — a tracker lookup must never block or break a review.
    """
    try:
        linear = config.linear
        if not linear.enabled:
            return []
        api_key = os.environ.get(str(linear.api_key_env), "").strip()
        if not api_key:
            return []
        identifiers = issue_identifiers_for_pr(pr_info, linear.team_keys)
        if not identifiers:
            return []
        return await LinearClient(api_key, str(linear.api_url)).fetch_issues(identifiers)
    except Exception as exc:  # noqa: BLE001 — never let ticket lookup break a review
        logger.warning("Linear issue resolution failed: %s", exc)
        return []


def format_issues_context(issues: list[LinkedIssue]) -> str:
    """Render linked issues as prompt context for the reviewer."""
    if not issues:
        return ""
    lines = [
        "## Linked Issues",
        "",
        "This PR references the tracker issues below. Check the diff against each "
        "issue's intent and acceptance criteria. Flag an inline comment when the "
        "change contradicts the ticket, leaves a stated requirement unmet, or "
        "implements something the ticket did not ask for. Do not restate the "
        "ticket back to the author.",
        "",
    ]
    for issue in issues:
        header = f"### {issue.identifier}"
        if issue.title:
            header += f" — {issue.title}"
        lines.append(header)
        if issue.state:
            lines.append(f"- **State:** {issue.state}")
        if issue.url:
            lines.append(f"- **URL:** {issue.url}")
        if issue.description:
            lines.append("")
            lines.append(issue.description)
        lines.append("")
    return "\n".join(lines).rstrip()
