"""Shared data models for Mira."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

WALKTHROUGH_MARKER = "<!-- mira-walkthrough -->"


class FileChangeType(enum.Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class Severity(enum.IntEnum):
    """Review comment severity, ordered from most to least severe."""

    BLOCKER = 4
    WARNING = 3
    SUGGESTION = 2
    NITPICK = 1

    @classmethod
    def from_str(cls, value: str) -> Severity:
        mapping = {
            "blocker": cls.BLOCKER,
            "critical": cls.BLOCKER,
            "error": cls.BLOCKER,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "suggestion": cls.SUGGESTION,
            "suggest": cls.SUGGESTION,
            "nitpick": cls.NITPICK,
            "nit": cls.NITPICK,
            "style": cls.NITPICK,
        }
        normalized = value.strip().lower()
        if normalized in mapping:
            return mapping[normalized]
        return cls.SUGGESTION

    @property
    def emoji(self) -> str:
        return {
            Severity.BLOCKER: "\U0001f6d1",  # stop sign
            Severity.WARNING: "\u26a0\ufe0f",  # warning
            Severity.SUGGESTION: "\U0001f4a1",  # light bulb
            Severity.NITPICK: "\U0001f4ac",  # speech bubble
        }[self]


@dataclass
class HunkInfo:
    """A single diff hunk within a file."""

    source_start: int
    source_length: int
    target_start: int
    target_length: int
    content: str


@dataclass
class FileDiff:
    """Parsed diff for a single file."""

    path: str
    change_type: FileChangeType
    hunks: list[HunkInfo] = field(default_factory=list)
    language: str = ""
    old_path: str | None = None
    is_binary: bool = False
    added_lines: int = 0
    deleted_lines: int = 0

    @property
    def total_changes(self) -> int:
        return self.added_lines + self.deleted_lines


@dataclass
class PatchSet:
    """A collection of file diffs representing a PR's changes."""

    files: list[FileDiff] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_additions(self) -> int:
        return sum(f.added_lines for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deleted_lines for f in self.files)


def build_review_stats(comments: list[ReviewComment]) -> dict[Severity, int]:
    """Count review comments grouped by severity.

    Returns a mapping of severity → count, only including severities with > 0 comments.
    """
    counts: dict[Severity, int] = {}
    for c in comments:
        counts[c.severity] = counts.get(c.severity, 0) + 1
    return counts


@dataclass
class KeyIssue:
    """A critical issue highlighted for human reviewers."""

    issue: str
    path: str
    line: int


@dataclass
class ReviewComment:
    """A single review comment to post."""

    path: str
    line: int
    end_line: int | None
    severity: Severity
    category: str
    title: str
    body: str
    confidence: float
    suggestion: str | None = None
    agent_prompt: str | None = None
    # Verbatim diff snippet used by self-critique; stripped before posting.
    existing_code: str = ""
    # Which pipeline pass produced this ("main", "security", or "osv") — lets eval
    # artifacts attribute FP share per pass. Not posted anywhere.
    source_pass: str = "main"


_SEVERITY_NAME: dict[Severity, str] = {
    Severity.BLOCKER: "blocker",
    Severity.WARNING: "warning",
    Severity.SUGGESTION: "suggestion",
    Severity.NITPICK: "nitpick",
}

_CHANGE_TYPE_NAME: dict[FileChangeType, str] = {
    FileChangeType.ADDED: "Added",
    FileChangeType.MODIFIED: "Modified",
    FileChangeType.DELETED: "Deleted",
    FileChangeType.RENAMED: "Renamed",
}

_CRITERION_GLYPH: dict[str, str] = {
    "met": "\u2705",
    "unmet": "\u274c",
    "unclear": "\u26a0\ufe0f",
}

# Cap per verdict section so a noisy PR can't turn the walkthrough into a wall
# of text; the inline comments remain the complete list.
_VERDICT_LIST_LIMIT = 10

# Cap the collapsed "What changed" list so a very large PR stays skimmable.
_CHANGES_DISPLAY_LIMIT = 15

# Cap the deterministic change-map diagram so it stays readable on large PRs.
_CHANGE_MAP_LIMIT = 12

# Canonical verdict labels. The walkthrough headline, the review-body verdict
# row, the posted review event, and the GitHub check-run conclusion all key off
# these, so one review can never show two different verdicts.
VERDICT_APPROVE = "Looks good to merge"
VERDICT_NEEDS_REVIEW = "Needs review"
VERDICT_REQUEST_CHANGES = "Request changes"


def _format_stats_breakdown(stats: dict[Severity, int]) -> str:
    """Format severity counts as a parenthetical breakdown, e.g. ' (1 blocker, 2 warnings)'."""
    labels = {
        Severity.BLOCKER: "blocker",
        Severity.WARNING: "warning",
        Severity.SUGGESTION: "suggestion",
        Severity.NITPICK: "nitpick",
    }
    items: list[str] = []
    for sev in (Severity.BLOCKER, Severity.WARNING, Severity.SUGGESTION, Severity.NITPICK):
        count = stats.get(sev, 0)
        if count:
            name = labels[sev]
            items.append(f"{sev.emoji} {count} {name}{'s' if count != 1 else ''}")
    return f" ({', '.join(items)})" if items else ""


def _format_finding_lines(comments: list[ReviewComment]) -> list[str]:
    """Render findings as ``- `path:line` — Title (severity)`` bullets."""
    lines: list[str] = []
    for c in comments[:_VERDICT_LIST_LIMIT]:
        title = c.title.strip() or "Issue"
        severity = _SEVERITY_NAME.get(c.severity, "")
        suffix = f" *({severity})*" if severity else ""
        lines.append(f"- `{c.path}:{c.line}` — {title}{suffix}")
    remaining = len(comments) - _VERDICT_LIST_LIMIT
    if remaining > 0:
        lines.append(f"- _…and {remaining} more (see inline comments)_")
    return lines


@dataclass
class WalkthroughConfidenceScore:
    """Confidence score for merge readiness."""

    score: int
    label: str
    reason: str


@dataclass
class Verdict:
    """Merge recommendation derived from the findings that actually got posted.

    The walkthrough headline and the "required changes" list are computed from
    the final inline comments, so a "Request changes" verdict always names what
    must change instead of leaving the reader with a bare label.
    """

    label: str
    emoji: str
    blockers: list[ReviewComment] = field(default_factory=list)
    warnings: list[ReviewComment] = field(default_factory=list)
    optional: list[ReviewComment] = field(default_factory=list)
    unmet_criteria: list[TicketCriterion] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.blockers or self.warnings or self.unmet_criteria)


def derive_verdict(
    comments: list[ReviewComment] | None,
    confidence_score: WalkthroughConfidenceScore | None = None,
    criteria: list[TicketCriterion] | None = None,
) -> Verdict:
    """Derive a merge verdict from the findings that survived filtering.

    Blockers — including any unmet linked-ticket requirement — force "Request
    changes"; warnings (or a low confidence score) give "Needs review";
    anything else is safe to merge. Each bucket keeps the comments themselves
    so callers can list exactly what has to change.
    """
    filed = comments or []
    blockers = [c for c in filed if c.severity == Severity.BLOCKER]
    warnings = [c for c in filed if c.severity == Severity.WARNING]
    optional = [c for c in filed if c.severity <= Severity.SUGGESTION]
    unmet = [c for c in (criteria or []) if c.is_unmet]

    if blockers or unmet:
        label, emoji = VERDICT_REQUEST_CHANGES, "\U0001f6d1"
    elif warnings or (confidence_score is not None and confidence_score.score <= 2):
        label, emoji = VERDICT_NEEDS_REVIEW, "\u26a0\ufe0f"
    else:
        label, emoji = VERDICT_APPROVE, "\u2705"

    return Verdict(
        label=label,
        emoji=emoji,
        blockers=blockers,
        warnings=warnings,
        optional=optional,
        unmet_criteria=unmet,
    )


def derive_review_verdict(result: ReviewResult) -> Verdict:
    """The single verdict every surface reports for a finished review.

    Findings decide it first. A linked ticket Mira could not read is the one
    other blocker: the ticket's acceptance criteria are part of what was asked
    for, so an unverifiable ticket can never be approved — it downgrades an
    otherwise-clean review to "Needs review" with the reason shown, instead of
    silently reporting a bare "no findings".

    Findings already posted and still open count too (``outstanding_comments``):
    a re-review that finds nothing new must not upgrade a PR whose blocker is
    still unresolved.
    """
    confidence = result.walkthrough.confidence_score if result.walkthrough else None
    filed = [*result.comments, *result.outstanding_comments]
    verdict = derive_verdict(filed, confidence, result.ticket_criteria)
    if not verdict.has_findings and result.linear_lookup_status == "unavailable":
        return Verdict(label=VERDICT_NEEDS_REVIEW, emoji="\u26a0\ufe0f")
    return verdict


def ticket_unverified_note(result: ReviewResult) -> str:
    """Why a referenced ticket could not be graded, or ``""`` when it could."""
    if result.linear_lookup_status != "unavailable":
        return ""
    identifiers = ", ".join(result.linear_issue_ids) or "the linked ticket"
    detail = f" ({result.linear_lookup_detail})" if result.linear_lookup_detail else ""
    return (
        f"Could not read {identifiers}{detail}, so its acceptance criteria are "
        "unverified — this review is not an approval."
    )


def documentation_paths(paths: list[str]) -> list[str]:
    """Paths that look like documentation, for the review status block."""
    return [
        path
        for path in paths
        if path.lower().endswith((".md", ".mdx", ".rst"))
        or path.lower().startswith(("docs/", "documentation/"))
    ]


def linear_ticket_status(result: ReviewResult) -> tuple[str, bool]:
    """Ticket row for the review status block: text + whether it verified."""
    if result.linear_lookup_status == "not_linked":
        return "N/A — No linked Linear ticket found", True
    if result.linear_lookup_status != "loaded":
        identifiers = ", ".join(result.linear_issue_ids)
        suffix = f" ({identifiers})" if identifiers else ""
        reason = f" — {result.linear_lookup_detail}" if result.linear_lookup_detail else ""
        return f"Unknown — Linear ticket could not be checked{suffix}{reason}", False

    links: list[str] = []
    for index, identifier in enumerate(result.linear_issue_ids):
        if index < len(result.linear_issue_urls):
            links.append(f"[{identifier}]({result.linear_issue_urls[index]})")
        else:
            links.append(identifier)
    linked = ", ".join(links)
    unmet = [c for c in result.ticket_criteria if c.is_unmet]
    if unmet:
        count = len(unmet)
        return (
            f"No — {linked}; {count} acceptance criterion{'s' if count != 1 else ''} unmet",
            False,
        )
    # This row grades the ticket, not the code: code findings are the verdict's
    # job (and the earlier "blocking review findings remain" wording made a
    # fully-graded ticket read like an unverified one).
    total = len(result.ticket_criteria)
    if total:
        met = len([c for c in result.ticket_criteria if c.status == "met"])
        unclear = total - met
        if unclear:
            return (
                f"Partly — {linked}; {met} of {total} criteria met, "
                f"{unclear} unclear from the diff",
                True,
            )
        return f"Yes — {linked}; all {total} acceptance criteria met", True
    return f"Yes — {linked}; no explicit acceptance criteria in the ticket", True


@dataclass
class LinkedIssue:
    """A tracker issue (e.g. Linear) referenced by the pull request."""

    identifier: str
    title: str = ""
    url: str = ""
    state: str = ""
    description: str = ""
    source: str = "linear"
    # Explicit acceptance criteria parsed from the ticket (markdown checkboxes
    # or an "Acceptance Criteria" section), if any.
    criteria: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    priority: str = ""
    assignee: str = ""
    # Recent comment bodies and child-issue summaries, for review context.
    comments: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)


@dataclass
class TicketCriterion:
    """One requirement from a linked ticket, graded against the PR diff."""

    issue: str
    criterion: str
    # "met" | "unmet" | "unclear" — unmet forces a "Request changes" verdict.
    status: str = "unclear"
    evidence: str = ""

    @property
    def is_unmet(self) -> bool:
        return self.status == "unmet"


@dataclass
class WalkthroughEffort:
    """Review effort estimate for a PR."""

    level: int
    label: str
    minutes: int


@dataclass
class WalkthroughFileEntry:
    """A single file entry in the walkthrough summary."""

    path: str
    change_type: FileChangeType
    description: str
    group: str = ""


@dataclass
class WalkthroughResult:
    """Result of the PR walkthrough generation."""

    summary: str = ""
    file_changes: list[WalkthroughFileEntry] = field(default_factory=list)
    effort: WalkthroughEffort | None = None
    confidence_score: WalkthroughConfidenceScore | None = None
    sequence_diagram: str | None = None

    def to_markdown(
        self,
        bot_name: str = "miracodeai",
        review_stats: dict[Severity, int] | None = None,
        existing_issues: int = 0,
        blast_radius: list[dict] | None = None,
        reviewed_files: int = 0,
        total_comments: int = 0,
        key_issues: list[KeyIssue] | None = None,
        in_progress: bool = False,
        skipped_paths: list[str] | None = None,
        total_paths: list[str] | None = None,
        index_was_empty: bool = False,
        dashboard_url: str = "",
        overlaps: list[OverlapFinding] | None = None,
        failure_notice: str | None = None,
        comments: list[ReviewComment] | None = None,
        linked_issues: list[LinkedIssue] | None = None,
        require_issue: bool = False,
        additions: int = 0,
        deletions: int = 0,
        ticket_criteria: list[TicketCriterion] | None = None,
        verdict: Verdict | None = None,
        verdict_note: str = "",
        status_rows: list[tuple[str, str]] | None = None,
        outstanding_count: int = 0,
    ) -> str:
        """Render as a markdown PR comment."""
        parts = [WALKTHROUGH_MARKER, "## Mira PR Walkthrough", ""]
        parts.append(self.summary)

        # "At a glance" line — how big the review is and how long it should
        # take. The model already produces the effort estimate; render it
        # instead of dropping it on the floor.
        if self.effort and not in_progress and not failure_notice:
            effort = self.effort
            parts.append("")
            line = f"\u23f1\ufe0f **Estimated review effort:** {effort.label} ({effort.level}/5)"
            if effort.minutes:
                line += f" \u00b7 ~{effort.minutes} min"
            parts.append(line)

        # The verdict and the changes it demands come first — a reader should
        # never have to open a collapsed block to learn why a PR needs work.
        # Suppressed during the in-progress render (findings aren't known yet)
        # and on failure (a "looks good" next to a failure notice is wrong).
        if not in_progress and not failure_notice:
            verdict_lines = self._render_verdict(
                comments, key_issues, ticket_criteria, verdict, verdict_note, outstanding_count
            )
            if verdict_lines:
                parts.append("")
                parts.extend(verdict_lines)

        if self.sequence_diagram:
            diagram = self.sequence_diagram.strip()
            # _sanitize_mermaid has already quoted labels with dots/slashes;
            # re-quoting here would reintroduce the nested-quote bug.
            if diagram and any(
                diagram.startswith(k) for k in ("graph ", "flowchart ", "sequenceDiagram")
            ):
                parts.append("")
                parts.append("```mermaid")
                parts.append(diagram)
                parts.append("```")
        else:
            # No LLM diagram: fall back to the deterministic change map so every
            # walkthrough still shows how the change fits together.
            change_map = self._render_change_map()
            if change_map:
                parts.append("")
                parts.extend(change_map)

        issue_lines = self._render_linked_issues(linked_issues, require_issue)
        if issue_lines:
            parts.append("")
            parts.extend(issue_lines)

        criteria_lines = self._render_ticket_criteria(ticket_criteria)
        if criteria_lines:
            parts.append("")
            parts.extend(criteria_lines)

        changes_lines = self._render_changes()
        if changes_lines:
            parts.append("")
            parts.extend(changes_lines)

        if overlaps:
            _kind_label = {
                "merge_conflict": "merge-conflict risk",
                "duplicate_effort": "duplicate effort",
                "both": "duplicate effort + merge-conflict risk",
            }
            parts.append("")
            parts.append(
                "> **⚠️ Potential overlap with other open PRs** — these may be stepping on this one:"
            )
            parts.append(">")
            for ov in overlaps:
                label = _kind_label.get(ov.kind, ov.kind)
                link = f"[#{ov.pr_number}]({ov.url})" if ov.url else f"#{ov.pr_number}"
                line = f"> - {link} ({label}) — {ov.reason}"
                if ov.shared_files:
                    shown = ", ".join(f"`{p}`" for p in ov.shared_files[:3])
                    if len(ov.shared_files) > 3:
                        shown += f" +{len(ov.shared_files) - 3} more"
                    line += f" Shared: {shown}"
                parts.append(line)
            parts.append("")

        if blast_radius:
            parts.append("")
            total_refs = sum(len(e.get("files", [])) for e in blast_radius)
            repo_count = len(blast_radius)
            header = f"{'repository' if repo_count == 1 else 'repositories'}"

            parts.append(
                f"> **Blast Radius** \u2014 {repo_count} dependent {header}, {total_refs} total references"
            )
            parts.append(">")
            for entry in blast_radius:
                repo = entry.get("repo", "")
                files = entry.get("files", [])
                parts.append(
                    f"> `{repo}` \u2014 {len(files)} reference{'s' if len(files) != 1 else ''}"
                )
            parts.append("")

        if in_progress:
            parts.append("")
            parts.append("*\u23f3 Code review in progress\u2026*")
        else:
            stats_parts: list[str] = []
            if reviewed_files:
                stats_parts.append(
                    f"{reviewed_files} file{'s' if reviewed_files != 1 else ''} reviewed"
                )
            if additions or deletions:
                diffstat = f"+{additions} \u2212{deletions}"
                stats_parts.append(f"`{diffstat}`")
            if total_comments:
                comment_detail = _format_stats_breakdown(review_stats) if review_stats else ""
                stats_parts.append(
                    f"{total_comments} comment{'s' if total_comments != 1 else ''}{comment_detail}"
                )
            if existing_issues:
                stats_parts.append(
                    f"{existing_issues} unresolved thread{'s' if existing_issues != 1 else ''}"
                )
            if stats_parts:
                separator = " \u00b7 "
                parts.append("")
                parts.append(f"*{separator.join(stats_parts)}*")

        # CI, docs, and ticket verification were a second comment (the review
        # body) until they moved here — the walkthrough is the one summary, so
        # the review itself can stay empty and the PR gets a single comment.
        if status_rows and not in_progress and not failure_notice:
            parts.append("")
            parts.append("### Review status")
            parts.append("")
            for label, value in status_rows:
                parts.append(f"- **{label}:** {value}")

        if skipped_paths and not in_progress:
            total = len(total_paths) if total_paths else (reviewed_files + len(skipped_paths))
            shown = min(8, len(skipped_paths))
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append(f"### \ud83d\udccb Reviewed {reviewed_files} of {total} files")
            parts.append("")
            parts.append(
                "This PR is large enough that some files were skipped to keep the "
                "review focused on the highest-priority changes. To review the rest, "
                f"comment `@{bot_name} review-rest` on this PR."
            )
            parts.append("")
            parts.append("**Skipped:**")
            for p in skipped_paths[:shown]:
                parts.append(f"- `{p}`")
            if len(skipped_paths) > shown:
                parts.append(f"- _\u2026and {len(skipped_paths) - shown} more_")

        if index_was_empty and not in_progress:
            parts.append("")
            parts.append("---")
            parts.append("")
            link = f"[Mira dashboard]({dashboard_url})" if dashboard_url else "the Mira dashboard"
            parts.append(
                f"> 💡 **This review will be more accurate after indexing.** "
                f"This repo hasn't been indexed yet, so the review is based on "
                f"the diff plus on-demand file lookups. Visit {link} to index "
                f"this repo — Mira will then know about callers, dependents, "
                f"and cross-repo impact."
            )

        if failure_notice:
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append(
                "<details>\n<summary><b>❌ Review failed</b> — click for details</summary>\n"
            )
            parts.append("")
            parts.append(failure_notice)
            parts.append("")
            parts.append("</details>")

        parts.append("")
        parts.append("---")
        parts.append(
            f"> Comment `@{bot_name} help` to get the list of available commands and usage tips."
        )

        return "\n".join(parts)

    def _render_change_map(self) -> list[str]:
        """Deterministic Mermaid map of the changed files, grouped by cohort.

        Used when the model produced no sequence diagram, so the walkthrough
        always shows how the pieces relate. Capped so a 100-file PR can't
        produce an unreadable diagram.
        """
        if not self.file_changes:
            return []

        entries = self.file_changes[:_CHANGE_MAP_LIMIT]
        group_ids: dict[str, str] = {}
        lines = ["flowchart LR", '  pr["Pull request"]']
        for index, entry in enumerate(entries):
            group = entry.group.strip() or "Changed files"
            if group not in group_ids:
                group_id = f"g{len(group_ids)}"
                group_ids[group] = group_id
                safe_group = " ".join(group.split()).replace('"', "'")
                lines.append(f'  {group_id}["{safe_group}"]')
                lines.append(f"  pr --> {group_id}")
            file_id = f"f{index}"
            safe_path = entry.path.replace('"', "'")
            lines.append(f'  {file_id}["{safe_path}"]')
            lines.append(f"  {group_ids[group]} --> {file_id}")
        remaining = len(self.file_changes) - len(entries)
        if remaining:
            lines.append(f'  more["+{remaining} more files"]')
            lines.append("  pr --> more")
        return ["```mermaid", *lines, "```"]

    def _render_verdict(
        self,
        comments: list[ReviewComment] | None,
        key_issues: list[KeyIssue] | None,
        ticket_criteria: list[TicketCriterion] | None = None,
        verdict: Verdict | None = None,
        note: str = "",
        outstanding_count: int = 0,
    ) -> list[str]:
        """Render the verdict headline plus exactly what must change.

        The headline is derived from the final findings rather than the model's
        free-form label, so a "Request changes" verdict is always accompanied by
        the blocker/warning list that justifies it. Callers that already derived
        the review verdict pass it in so the headline, the review-body verdict
        row, and the check run can never disagree.
        """
        cs = self.confidence_score
        if verdict is None:
            verdict = derive_verdict(comments, cs, ticket_criteria)

        # Nothing to say: no score, no findings, no key issues, no criteria.
        has_findings = bool(
            verdict.blockers or verdict.warnings or verdict.optional or verdict.unmet_criteria
        )
        if cs is None and not has_findings and not key_issues and not ticket_criteria and not note:
            return []

        lines = [f"## Verdict: {verdict.emoji} {verdict.label}", ""]
        if cs is not None:
            filled = "\u25c9" * cs.score  # ◉
            empty = "\u25cb" * (5 - cs.score)  # ○
            line = f"{filled}{empty} **{cs.score}/5 confidence**"
            reason = cs.reason.strip()
            if reason:
                line += f" — {reason}"
            lines.append(line)
            lines.append("")
        if note:
            lines.append(f"> {note}")
            lines.append("")
        if outstanding_count:
            lines.append(
                f"> **{outstanding_count} finding"
                f"{'s' if outstanding_count != 1 else ''} below "
                f"{'are' if outstanding_count != 1 else 'is'} still open from an earlier "
                "review on this PR** — resolve the thread once it's fixed."
            )
            lines.append("")

        if verdict.blockers:
            lines.append("**Blockers — must fix before merge:**")
            lines.append("")
            lines.extend(_format_finding_lines(verdict.blockers))
            lines.append("")
        if verdict.unmet_criteria:
            lines.append("**Ticket requirements not met:**")
            lines.append("")
            for c in verdict.unmet_criteria:
                line = f"- `{c.issue}` — {c.criterion}"
                if c.evidence:
                    line += f" — {c.evidence}"
                lines.append(line)
            lines.append("")
        if verdict.warnings:
            lines.append("**Warnings — should fix before merge:**")
            lines.append("")
            lines.extend(_format_finding_lines(verdict.warnings))
            lines.append("")

        # Fall back to the model's key issues when no inline survived filtering
        # (older callers pass only key_issues).
        if not verdict.has_findings and key_issues:
            lines.append("**Key files to review:**")
            lines.append("")
            for ki in key_issues[:_VERDICT_LIST_LIMIT]:
                lines.append(f"- `{ki.path}:{ki.line}` — {ki.issue}")
            lines.append("")

        if verdict.optional:
            lines.append("<details>")
            lines.append(
                f"<summary><b>Optional suggestions ({len(verdict.optional)})</b></summary>"
            )
            lines.append("")
            lines.extend(_format_finding_lines(verdict.optional))
            lines.append("")
            lines.append("</details>")
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _render_linked_issues(
        self,
        linked_issues: list[LinkedIssue] | None,
        require_issue: bool,
    ) -> list[str]:
        """Render the tracker issues this PR references, if any."""
        issues = linked_issues or []
        if not issues and not require_issue:
            return []

        lines = ["### Linked issues", ""]
        if not issues:
            lines.append(
                "- _No linked issue found in the PR title, description, or branch. "
                "Link the tracked ticket so reviewers can verify intent._"
            )
            return lines

        for issue in issues:
            label = f"[{issue.identifier}]({issue.url})" if issue.url else f"`{issue.identifier}`"
            line = f"- {label}"
            if issue.title:
                line += f" — {issue.title}"
            if issue.state:
                line += f" *({issue.state})*"
            lines.append(line)
        return lines

    def _render_ticket_criteria(
        self,
        ticket_criteria: list[TicketCriterion] | None,
    ) -> list[str]:
        """Render every linked-ticket requirement with its verification status.

        Unmet requirements are never hidden: a ticket reference always produces
        a full checklist, and a criterion the verification pass could not grade
        shows as "unclear" rather than silently passing.
        """
        if not ticket_criteria:
            return []
        by_issue: dict[str, list[TicketCriterion]] = {}
        for c in ticket_criteria:
            by_issue.setdefault(c.issue, []).append(c)

        lines = [
            "### Ticket acceptance criteria",
            "",
            "✅ met · ❌ unmet · ⚠️ unclear — not verifiable from this diff",
            "",
        ]
        for issue, criteria in by_issue.items():
            lines.append(f"**{issue}**")
            lines.append("")
            for c in criteria:
                glyph = _CRITERION_GLYPH.get(c.status, "\u2022")
                line = f"- {glyph} {c.criterion}"
                if c.evidence:
                    line += f" — {c.evidence}"
                lines.append(line)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _render_changes(self) -> list[str]:
        """Render per-file change descriptions grouped into logical cohorts.

        Collapsed behind a ``<details>`` so a large PR's file list doesn't
        dominate the comment — the summary and verdict stay above the fold —
        and capped so a 100-file PR can't produce a wall of text.
        """
        if not self.file_changes:
            return []
        grouped: dict[str, list[WalkthroughFileEntry]] = {}
        for entry in self.file_changes:
            grouped.setdefault(entry.group or "Other changes", []).append(entry)

        total = len(self.file_changes)
        lines = [
            "<details>",
            f"<summary><b>What changed</b> — {total} file{'s' if total != 1 else ''}</summary>",
            "",
        ]
        rendered = 0
        skipped = 0
        for group, entries in grouped.items():
            bullets: list[str] = []
            for entry in entries:
                if rendered >= _CHANGES_DISPLAY_LIMIT:
                    skipped += 1
                    continue
                change = _CHANGE_TYPE_NAME.get(entry.change_type, "Modified")
                line = f"- **{change}** `{entry.path}`"
                description = entry.description.strip()
                if description:
                    line += f" — {description}"
                bullets.append(line)
                rendered += 1
            if bullets:
                lines.append(f"**{group}**")
                lines.append("")
                lines.extend(bullets)
                lines.append("")
        if skipped:
            lines.append(f"_…and {skipped} more file{'s' if skipped != 1 else ''}_")
            lines.append("")
        lines.append("</details>")
        return lines


@dataclass
class ThreadDecision:
    """Per-thread resolution decision from dry-run."""

    thread_id: str
    path: str
    line: int
    body: str
    fixed: bool


@dataclass
class ReviewResult:
    """The complete result of a review."""

    comments: list[ReviewComment] = field(default_factory=list)
    key_issues: list[KeyIssue] = field(default_factory=list)
    summary: str = ""
    reviewed_files: int = 0
    skipped_reason: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    walkthrough: WalkthroughResult | None = None
    thread_decisions: list[ThreadDecision] = field(default_factory=list)
    # Line counts across the files actually reviewed, for the walkthrough header.
    additions: int = 0
    deletions: int = 0
    # Linked-ticket requirements graded against the diff. Any unmet criterion
    # forces a "Request changes" verdict.
    ticket_criteria: list[TicketCriterion] = field(default_factory=list)
    # Surfaced in the walkthrough banner so @miracodeai review-rest can target the rest.
    reviewed_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    total_paths: list[str] = field(default_factory=list)
    linear_issue_ids: list[str] = field(default_factory=list)
    linear_issue_urls: list[str] = field(default_factory=list)
    linear_lookup_status: str = "not_linked"
    # Why a referenced ticket could not be graded (missing key, API error, not
    # found), shown next to the "Unknown" verdict so it is actionable.
    linear_lookup_detail: str = ""
    # Mira's own review threads that are still open from earlier rounds, rebuilt
    # as findings. A re-review doesn't re-draft what it already posted (the
    # prompt asks it not to repeat itself and drop_already_posted removes
    # overlaps), so without these the verdict would soften while a blocker is
    # still open on the PR.
    outstanding_comments: list[ReviewComment] = field(default_factory=list)
    # Diagnostic trail: per-chunk draft counts and every comment dropped by a
    # filter/critique stage, so a benchmark run can show whether a missed
    # finding was never drafted or drafted-then-dropped. Not posted anywhere.
    audit: list[dict] = field(default_factory=list)


@dataclass
class PRInfo:
    """Metadata about a pull request."""

    title: str
    description: str
    base_branch: str
    head_branch: str
    url: str
    number: int
    owner: str
    repo: str
    # Round 2+ reviews diff against last_reviewed_sha → head_sha; empty falls back to full diff.
    head_sha: str = ""
    # Hosting platform ("github" / "gitlab") — scopes per-PR review progress.
    platform: str = "github"
    # Platform login of the PR author; used to attribute review-quality stats
    # and surfaced (with avatar) in the activity dashboard.
    author: str = ""
    author_avatar_url: str = ""


@dataclass
class OpenPRRef:
    """Lightweight handle on another open PR in the same repo.

    Built from the GitHub "list pull requests" API — just enough to decide
    whether the PR is worth comparing against the one under review (without
    fetching its full diff up front).
    """

    number: int
    title: str
    body: str
    head_sha: str
    author: str
    draft: bool = False
    base_ref: str = ""
    head_ref: str = ""
    url: str = ""


@dataclass
class PRFingerprint:
    """A compact signature of a PR's changes, cached per repo.

    Populated for a PR the moment Mira reviews it (the diff is already in
    hand), so a later review of a *different* PR can compare against it with a
    cheap DB read instead of re-fetching this PR's files from GitHub.
    """

    pr_number: int
    head_sha: str
    title: str
    body: str
    paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    updated_at: float = 0.0


@dataclass
class OverlapFinding:
    """A confirmed overlap between the PR under review and another open PR."""

    pr_number: int
    url: str
    title: str
    # 'merge_conflict' (touch the same code) | 'duplicate_effort' (same goal)
    # | 'both'.
    kind: str
    reason: str
    confidence: float
    shared_files: list[str] = field(default_factory=list)


@dataclass
class UnresolvedThread:
    """An unresolved review thread authored by the bot."""

    thread_id: str
    path: str
    line: int
    body: str
    is_outdated: bool = False


@dataclass
class BotThreadRecord:
    """A review thread authored by the bot, resolved or not."""

    thread_id: str
    path: str
    line: int
    body: str
    is_resolved: bool
    is_outdated: bool = False


@dataclass
class HumanReviewComment:
    """A review comment on a PR authored by a human (not the bot)."""

    path: str
    line: int
    body: str
    author: str


@dataclass
class FileHistoryEntry:
    """A commit that previously touched a file. Used by decision archaeology
    to give the review LLM context on why code exists before suggesting it
    be changed or removed."""

    sha: str
    message: str
    author: str
    date: str  # ISO-8601 timestamp from the GitHub API


@dataclass
class ReviewChunk:
    """A chunk of files that fits within a single LLM context window."""

    files: list[FileDiff] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class FeedbackEvent:
    """A recorded feedback signal on a review comment."""

    id: int
    pr_number: int
    pr_url: str
    comment_path: str
    comment_line: int
    comment_category: str
    comment_severity: str
    comment_title: str
    signal: str  # 'rejected' | 'accepted'
    actor: str
    created_at: float = 0.0


@dataclass
class LearnedRule:
    """A rule synthesised from accumulated feedback patterns."""

    id: int
    rule_text: str
    source_signal: str  # 'reject_pattern' | 'accept_pattern'
    category: str
    path_pattern: str  # e.g. 'tests/**' or '' for all
    sample_count: int
    active: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0
