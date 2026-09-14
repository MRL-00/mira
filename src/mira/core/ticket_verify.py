"""Verify that a PR satisfies every requirement of its linked tracker tickets.

Runs after the diff is fetched and before the walkthrough is finalized. For
each linked issue it grades every explicit acceptance criterion (and, when the
ticket states none, the requirements derived from its description) against the
diff. Anything the model does not address is surfaced as ``unclear`` rather
than silently passing, and any ``unmet`` criterion forces a "Request changes"
verdict.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mira.llm.response_parser import loads_lenient
from mira.llm.tool_schemas import SUBMIT_TICKET_VERIFICATION_TOOL
from mira.models import LinkedIssue, TicketCriterion

if TYPE_CHECKING:
    from mira.llm.base import LLMProviderProtocol

logger = logging.getLogger(__name__)

_VALID_STATUSES = {"met", "unmet", "unclear"}
_MAX_DIFF_CHARS = 60_000
_MAX_COMMENTS = 10


def _truncate_on_line(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    clipped = text[:limit]
    newline = clipped.rfind("\n")
    if newline > 0:
        clipped = clipped[:newline]
    return clipped.rstrip() + "\n… (diff truncated)"


def _render_issue(issue: LinkedIssue) -> str:
    lines = [f"### {issue.identifier}" + (f" — {issue.title}" if issue.title else "")]
    meta: list[str] = []
    if issue.state:
        meta.append(f"State: {issue.state}")
    if issue.priority:
        meta.append(f"Priority: {issue.priority}")
    if issue.labels:
        meta.append(f"Labels: {', '.join(issue.labels)}")
    if meta:
        lines.append("- " + " · ".join(meta))
    if issue.url:
        lines.append(f"- URL: {issue.url}")
    if issue.description:
        lines.append("")
        lines.append("Description:")
        lines.append(issue.description)
    if issue.criteria:
        lines.append("")
        lines.append("Acceptance criteria (every one MUST be graded):")
        lines.extend(f"- {c}" for c in issue.criteria)
    if issue.children:
        lines.append("")
        lines.append("Sub-issues:")
        lines.extend(f"- {c}" for c in issue.children)
    if issue.comments:
        lines.append("")
        lines.append("Recent ticket comments:")
        lines.extend(f"- {c}" for c in issue.comments[:_MAX_COMMENTS])
    return "\n".join(lines)


def _build_prompt(issues: list[LinkedIssue], diff_text: str) -> str:
    tickets = "\n\n".join(_render_issue(i) for i in issues)
    diff = _truncate_on_line(diff_text, _MAX_DIFF_CHARS)
    return (
        "You are verifying whether a pull request satisfies the linked tracker "
        "ticket(s). For every requirement, output exactly one entry with a status:\n\n"
        "- `met` — the diff demonstrably implements the requirement.\n"
        "- `unmet` — the diff shows it was missed, done incorrectly, or not "
        "implemented at all. Partial or incomplete work counts as unmet.\n"
        "- `unclear` — you genuinely cannot tell from the diff alone.\n\n"
        "Be strict. A requirement is only `met` when the diff actually implements "
        "it. Grade EVERY explicit acceptance criterion — do not omit any. When a "
        "ticket states no explicit criteria, derive the concrete requirements "
        "from its description and comments, then grade each one. Quote the "
        "requirement in `criterion` so the author can see exactly what was "
        "checked.\n\n"
        "## Linked tickets\n\n"
        f"{tickets}\n\n"
        "## Pull request diff\n\n"
        "```diff\n"
        f"{diff}\n"
        "```\n"
    )


def _normalize(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _covered(explicit: str, returned: list[TicketCriterion]) -> bool:
    """Whether the model produced a verdict for an explicit criterion."""
    target = _normalize(explicit)
    if not target:
        return True
    target_tokens = set(target.split())
    for c in returned:
        candidate = _normalize(c.criterion)
        if not candidate:
            continue
        if target in candidate or candidate in target:
            return True
        candidate_tokens = set(candidate.split())
        if target_tokens and candidate_tokens:
            overlap = len(target_tokens & candidate_tokens) / len(target_tokens)
            if overlap >= 0.6:
                return True
    return False


def _fallback_for_unassessed(issues: list[LinkedIssue]) -> list[TicketCriterion]:
    """Every explicit criterion as `unclear`, for when grading can't run."""
    return [
        TicketCriterion(issue=issue.identifier, criterion=c, status="unclear")
        for issue in issues
        for c in issue.criteria
    ]


async def verify_ticket_criteria(
    llm: LLMProviderProtocol,
    issues: list[LinkedIssue],
    diff_text: str,
) -> list[TicketCriterion]:
    """Grade every linked-ticket requirement against the diff.

    Best-effort: a failed or unparseable model call degrades to "unclear" for
    each explicit criterion so a ticket reference is never silently treated as
    satisfied.
    """
    if not issues or not diff_text.strip():
        return []

    prompt = _build_prompt(issues, diff_text)
    try:
        raw = await llm.complete_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=[SUBMIT_TICKET_VERIFICATION_TOOL],
            temperature=0.0,
        )
        data = loads_lenient(raw) if raw else None
    except Exception as exc:  # noqa: BLE001 — never let ticket verification break a review
        logger.warning("Ticket verification failed, marking criteria unclear: %s", exc)
        return _fallback_for_unassessed(issues)

    if not isinstance(data, dict):
        return _fallback_for_unassessed(issues)

    known_issues = {i.identifier for i in issues}
    criteria: list[TicketCriterion] = []
    for item in data.get("criteria") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("criterion") or "").strip()
        if not text:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status not in _VALID_STATUSES:
            status = "unclear"
        issue = str(item.get("issue") or "").strip()
        if issue not in known_issues:
            # A model that invents an identifier shouldn't drop the finding —
            # attach it to the first linked ticket.
            issue = issues[0].identifier
        criteria.append(
            TicketCriterion(
                issue=issue,
                criterion=text,
                status=status,
                evidence=str(item.get("evidence") or "").strip(),
            )
        )

    # Explicit criteria the model skipped must still be shown, never assumed met.
    for issue in issues:
        for explicit in issue.criteria:
            if not _covered(explicit, criteria):
                criteria.append(
                    TicketCriterion(
                        issue=issue.identifier,
                        criterion=explicit,
                        status="unclear",
                        evidence="not assessed by the verification pass",
                    )
                )

    unmet = sum(1 for c in criteria if c.status == "unmet")
    logger.info(
        "Ticket verification: %d criteria graded (%d unmet) across %d ticket(s)",
        len(criteria),
        unmet,
        len(issues),
    )
    return criteria
