"""Tests for walkthrough prompt builder, response parser, and markdown rendering."""

from __future__ import annotations

import json

import pytest

from mira.config import MiraConfig
from mira.llm.prompts.review import build_walkthrough_prompt
from mira.llm.response_parser import (
    convert_to_walkthrough_result,
    parse_walkthrough_response,
)
from mira.models import (
    VERDICT_NEEDS_REVIEW,
    WALKTHROUGH_MARKER,
    FileChangeType,
    FileDiff,
    HunkInfo,
    ReviewComment,
    ReviewResult,
    Severity,
    TicketCriterion,
    Verdict,
    WalkthroughConfidenceScore,
    WalkthroughEffort,
    WalkthroughFileEntry,
    WalkthroughResult,
    build_review_stats,
    derive_review_verdict,
    ticket_unverified_note,
)


class TestBuildWalkthroughPrompt:
    def _make_files(self) -> list[FileDiff]:
        return [
            FileDiff(
                path="src/utils.py",
                change_type=FileChangeType.ADDED,
                hunks=[
                    HunkInfo(
                        source_start=0,
                        source_length=0,
                        target_start=1,
                        target_length=5,
                        content="@@ -0,0 +1,5 @@\n+import os\n+def run(): pass",
                    )
                ],
                language="python",
                added_lines=5,
                deleted_lines=0,
            ),
            FileDiff(
                path="src/main.py",
                change_type=FileChangeType.MODIFIED,
                hunks=[
                    HunkInfo(
                        source_start=10,
                        source_length=3,
                        target_start=10,
                        target_length=5,
                        content=(
                            "@@ -10,3 +10,5 @@ class App:\n"
                            "     def start(self):\n+        debug=False"
                        ),
                    )
                ],
                language="python",
                added_lines=2,
                deleted_lines=0,
            ),
        ]

    def test_returns_two_messages(self):
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=MiraConfig(),
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_contains_file_metadata(self):
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=MiraConfig(),
        )
        system = messages[0]["content"]
        assert "src/utils.py" in system
        assert "src/main.py" in system
        assert "added" in system
        assert "modified" in system

    def test_includes_pr_title(self):
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=MiraConfig(),
            pr_title="Add utilities",
            pr_description="Some new helpers",
        )
        system = messages[0]["content"]
        assert "Add utilities" in system
        assert "Some new helpers" in system

    def test_sequence_diagram_flag(self):
        config = MiraConfig()
        config.review.walkthrough_sequence_diagram = True
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=config,
        )
        system = messages[0]["content"]
        assert "sequence_diagram" in system or "sequence diagram" in system.lower()
        # Template must instruct the LLM to use graph LR and avoid sequence diagrams
        assert "graph LR" in system
        assert "**Do NOT**" in system
        assert "null" in system  # instruction to omit when no interactions

    def test_hunk_headers_extracted(self):
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=MiraConfig(),
        )
        system = messages[0]["content"]
        assert "@@ -0,0 +1,5 @@" in system

    def test_diff_excerpts_in_user_message(self):
        messages = build_walkthrough_prompt(
            files=self._make_files(),
            config=MiraConfig(),
        )
        user = messages[1]["content"]
        # Actual changed code, not just filenames, so descriptions can be specific.
        assert "+import os" in user
        assert "+        debug=False" in user

    def test_diff_budget_truncates_excerpts(self):
        config = MiraConfig()
        config.review.walkthrough_diff_budget = 40
        messages = build_walkthrough_prompt(files=self._make_files(), config=config)
        user = messages[1]["content"]
        assert "diff truncated" in user

    def test_zero_budget_omits_excerpts(self):
        config = MiraConfig()
        config.review.walkthrough_diff_budget = 0
        messages = build_walkthrough_prompt(files=self._make_files(), config=config)
        user = messages[1]["content"]
        assert "+import os" not in user
        assert "diff truncated" not in user


class TestParseWalkthroughResponse:
    def test_basic_parse(self, sample_walkthrough_response_text: str):
        result = parse_walkthrough_response(sample_walkthrough_response_text)
        assert result.summary != ""
        assert len(result.change_groups) == 2
        assert result.change_groups[0].label == "Core"
        assert result.change_groups[0].files[0].path == "src/utils.py"
        assert result.change_groups[0].files[0].change_type == "added"

    def test_with_code_fences(self):
        raw = '```json\n{"summary": "test", "change_groups": []}\n```'
        result = parse_walkthrough_response(raw)
        assert result.summary == "test"

    def test_invalid_json_raises(self):
        from mira.exceptions import ResponseParseError

        with pytest.raises(ResponseParseError, match="not valid JSON"):
            parse_walkthrough_response("NOT JSON {{{")

    def test_non_object_raises(self):
        from mira.exceptions import ResponseParseError

        with pytest.raises(ResponseParseError, match="Expected JSON object"):
            parse_walkthrough_response("[1, 2, 3]")

    def test_with_sequence_diagram(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "sequence_diagram": "sequenceDiagram\n    A->>B: call",
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.sequence_diagram is not None
        assert "sequenceDiagram" in result.sequence_diagram

    def test_with_effort(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": {"level": 3, "label": "Moderate", "minutes": 20},
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.effort is not None
        assert result.effort.level == 3
        assert result.effort.label == "Moderate"
        assert result.effort.minutes == 20

    def test_without_effort(self):
        raw = json.dumps({"summary": "Changes", "change_groups": []})
        result = parse_walkthrough_response(raw)
        assert result.effort is None

    def test_leaked_arg_xml_in_object_fields(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": "level</arg_key><arg_value>2",
                "confidence_score": "score</arg_key><arg_value>4",
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert result.effort is not None
        assert result.effort.level == 2
        assert result.confidence_score is not None
        assert result.confidence_score.score == 4

    def test_full_arg_xml_fragment_rebuilt(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": (
                    "<arg_key>level</arg_key><arg_value>2</arg_value>"
                    "<arg_key>label</arg_key><arg_value>Trivial</arg_value>"
                ),
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.effort is not None
        assert result.effort.level == 2
        assert result.effort.label == "Trivial"

    def test_malformed_object_field_dropped(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": "some random garbage",
                "confidence_score": {"score": 1, "label": "x", "reason": "y"},
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert result.effort is None
        assert result.confidence_score is not None
        assert result.confidence_score.score == 1

    def test_trailing_tool_xml_truncated(self):
        raw = '{"summary": "Changes", "change_groups": []}</parameter></invoke>'
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"

    def test_skips_file_missing_path(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [
                    {
                        "label": "Core",
                        "files": [
                            {"path": "src/a.py", "change_type": "added"},
                            {"change_type": "modified", "description": "no path"},
                        ],
                    }
                ],
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert len(result.change_groups) == 1
        assert len(result.change_groups[0].files) == 1
        assert result.change_groups[0].files[0].path == "src/a.py"

    def test_skips_group_missing_label(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [
                    {"files": [{"path": "src/a.py"}]},
                    {"label": "Tests", "files": [{"path": "tests/b.py"}]},
                ],
            }
        )
        result = parse_walkthrough_response(raw)
        assert len(result.change_groups) == 1
        assert result.change_groups[0].label == "Tests"
        assert result.change_groups[0].files[0].path == "tests/b.py"

    def test_skips_non_dict_group(self):
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": ["not a group", {"label": "Core", "files": []}],
            }
        )
        result = parse_walkthrough_response(raw)
        assert len(result.change_groups) == 1
        assert result.change_groups[0].label == "Core"

    def test_salvage_failure_defaults_effort(self):
        """Leaked XML fragment that fails sub-model validation → effort drops to None (P3)."""
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": ("level</arg_key><arg_value>high"),
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert result.effort is None

    def test_salvage_failure_defaults_confidence_score(self):
        """Leaked XML fragment that fails sub-model validation → confidence_score drops to None (P3)."""
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "confidence_score": ("score</arg_key><arg_value>uncertain"),
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert result.confidence_score is None

    def test_malformed_dict_field_dropped(self):
        """Dict-form effort with invalid value fails LLMWalkthroughEffort validation → dropped to None."""
        raw = json.dumps(
            {
                "summary": "Changes",
                "change_groups": [],
                "effort": {"level": 3, "label": "Moderate", "minutes": "not_an_int"},
            }
        )
        result = parse_walkthrough_response(raw)
        assert result.summary == "Changes"
        assert result.effort is None


class TestConvertToWalkthroughResult:
    def test_basic_conversion(self, sample_walkthrough_response_text: str):
        parsed = parse_walkthrough_response(sample_walkthrough_response_text)
        result = convert_to_walkthrough_result(parsed)
        assert isinstance(result, WalkthroughResult)
        assert result.summary != ""
        assert len(result.file_changes) == 2
        assert result.file_changes[0].change_type == FileChangeType.ADDED
        assert result.file_changes[0].group == "Core"
        assert result.file_changes[1].change_type == FileChangeType.MODIFIED
        assert result.file_changes[1].group == "App Shell"

    def test_unknown_change_type_defaults_to_modified(self):
        from mira.llm.response_parser import (
            LLMWalkthroughChangeGroup,
            LLMWalkthroughFileChange,
            LLMWalkthroughResponse,
        )

        response = LLMWalkthroughResponse(
            summary="test",
            change_groups=[
                LLMWalkthroughChangeGroup(
                    label="Misc",
                    files=[
                        LLMWalkthroughFileChange(
                            path="foo.py", change_type="unknown_type", description="desc"
                        )
                    ],
                )
            ],
        )
        result = convert_to_walkthrough_result(response)
        assert result.file_changes[0].change_type == FileChangeType.MODIFIED
        assert result.file_changes[0].group == "Misc"

    def test_effort_conversion(self):
        raw = json.dumps(
            {
                "summary": "test",
                "change_groups": [],
                "effort": {"level": 2, "label": "Simple", "minutes": 10},
            }
        )
        parsed = parse_walkthrough_response(raw)
        result = convert_to_walkthrough_result(parsed)
        assert result.effort is not None
        assert result.effort.level == 2
        assert result.effort.label == "Simple"
        assert result.effort.minutes == 10

    def test_no_effort_conversion(self):
        raw = json.dumps({"summary": "test", "change_groups": []})
        parsed = parse_walkthrough_response(raw)
        result = convert_to_walkthrough_result(parsed)
        assert result.effort is None


class TestWalkthroughToMarkdown:
    def test_summary_rendered(self):
        result = WalkthroughResult(
            summary="Added new features.",
            file_changes=[
                WalkthroughFileEntry(
                    path="src/utils.py",
                    change_type=FileChangeType.ADDED,
                    description="New utils",
                    group="Core",
                ),
            ],
        )
        md = result.to_markdown()
        assert "## Mira PR Walkthrough" in md
        assert "Added new features." in md
        lines = md.split("\n")
        assert "---" in lines, "Expected separator '---' in markdown output"
        separator_idx = len(lines) - 1 - lines[::-1].index("---")
        footer_text = "\n".join(lines[separator_idx:])
        assert "@miracodeai help" in footer_text

    def test_with_sequence_diagram(self):
        result = WalkthroughResult(
            summary="Changes.",
            sequence_diagram="sequenceDiagram\n    A->>B: call",
        )
        md = result.to_markdown()
        assert "```mermaid" in md
        assert "sequenceDiagram" in md

    def test_no_files_no_table(self):
        result = WalkthroughResult(summary="Empty.")
        md = result.to_markdown()
        assert "### Changes" not in md
        assert "| File |" not in md

    def test_no_diagram_no_section(self):
        result = WalkthroughResult(summary="No diagram.")
        md = result.to_markdown()
        assert "### Sequence Diagram" not in md
        assert "```mermaid" not in md

    def test_change_map_rendered_without_sequence_diagram(self):
        """The deterministic map keeps a Mermaid diagram in every walkthrough."""
        result = WalkthroughResult(
            summary="Changes.",
            file_changes=[
                WalkthroughFileEntry("a.py", FileChangeType.MODIFIED, "x", "Core"),
                WalkthroughFileEntry("b.py", FileChangeType.ADDED, "y", "Tests"),
            ],
        )
        md = result.to_markdown()
        assert "```mermaid" in md
        assert 'pr["Pull request"]' in md
        assert 'f0["a.py"]' in md
        assert 'f1["b.py"]' in md

    def test_sequence_diagram_wins_over_change_map(self):
        result = WalkthroughResult(
            summary="Changes.",
            sequence_diagram="graph LR\n  a-->b",
            file_changes=[
                WalkthroughFileEntry("a.py", FileChangeType.MODIFIED, "x", "Core"),
            ],
        )
        md = result.to_markdown()
        assert md.count("```mermaid") == 1
        assert "a-->b" in md

    def test_with_confidence_score(self):
        from mira.models import WalkthroughConfidenceScore

        result = WalkthroughResult(
            summary="Changes.",
            confidence_score=WalkthroughConfidenceScore(
                score=4, label="Safe with minor fixes", reason="Looks good overall."
            ),
        )
        md = result.to_markdown()
        # The verdict and its rationale are visible, not hidden in a <details>.
        assert "## Verdict:" in md
        assert "4/5 confidence" in md
        assert "Looks good overall." in md

    def _comment(
        self,
        severity: Severity,
        path: str = "x.py",
        line: int = 1,
        title: str = "t",
    ):
        return ReviewComment(
            path=path,
            line=line,
            end_line=None,
            severity=severity,
            category="bug",
            title=title,
            body="b",
            confidence=0.9,
        )

    def test_verdict_lists_required_changes(self):
        from mira.models import WalkthroughConfidenceScore

        result = WalkthroughResult(
            summary="Changes.",
            confidence_score=WalkthroughConfidenceScore(2, "Request changes", "Unbounded loop."),
        )
        comments = [
            self._comment(Severity.BLOCKER, path="a.py", line=10, title="Unbounded retry loop"),
            self._comment(Severity.WARNING, path="b.py", line=3, title="Missing default"),
        ]
        md = result.to_markdown(comments=comments)
        assert "## Verdict: \U0001f6d1 Request changes" in md
        assert "Blockers — must fix before merge:" in md
        assert "`a.py:10` — Unbounded retry loop" in md
        assert "Warnings — should fix before merge:" in md
        assert "`b.py:3` — Missing default" in md

    def test_verdict_needs_review_for_warnings_only(self):
        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(comments=[self._comment(Severity.WARNING)])
        assert "## Verdict: \u26a0\ufe0f Needs review" in md
        assert "`x.py:1` — t" in md

    def test_verdict_derived_ignores_optimistic_label(self):
        from mira.models import WalkthroughConfidenceScore

        result = WalkthroughResult(
            summary="Changes.",
            confidence_score=WalkthroughConfidenceScore(5, "Safe to merge", "no risks"),
        )
        md = result.to_markdown(comments=[self._comment(Severity.BLOCKER)])
        # Findings win over the model's free-form label.
        assert "Request changes" in md
        assert "Safe to merge" not in md

    def test_verdict_suppressed_in_progress(self):
        result = WalkthroughResult(
            summary="Changes.",
            confidence_score=WalkthroughConfidenceScore(5, "Safe", "ok"),
        )
        md = result.to_markdown(in_progress=True)
        assert "## Verdict:" not in md

    def test_verdict_suppressed_on_failure(self):
        result = WalkthroughResult(
            summary="Changes.",
            confidence_score=WalkthroughConfidenceScore(5, "Safe", "ok"),
        )
        md = result.to_markdown(failure_notice="boom")
        assert "## Verdict:" not in md

    def test_passed_verdict_and_note_render(self):
        """An engine-derived verdict plus its reason drives the headline."""
        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(
            verdict=Verdict(label=VERDICT_NEEDS_REVIEW, emoji="\u26a0\ufe0f"),
            verdict_note="Could not read EPIC-1113 — no Linear API key set.",
        )
        assert "## Verdict: \u26a0\ufe0f Needs review" in md
        assert "> Could not read EPIC-1113 — no Linear API key set." in md

    def test_optional_suggestions_collapsed(self):
        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(comments=[self._comment(Severity.SUGGESTION)])
        assert "Optional suggestions (1)" in md
        assert "<details>" in md

    def test_changes_section_groups_files(self):
        result = WalkthroughResult(
            summary="Changes.",
            file_changes=[
                WalkthroughFileEntry("a.py", FileChangeType.ADDED, "New helper", "Core"),
                WalkthroughFileEntry("b.py", FileChangeType.MODIFIED, "Wire it up", "Core"),
                WalkthroughFileEntry("c.py", FileChangeType.MODIFIED, "Test it", "Tests"),
            ],
        )
        md = result.to_markdown()
        # Collapsed so a large file list doesn't dominate the comment.
        assert "<details>" in md
        assert "<summary><b>What changed</b> — 3 files</summary>" in md
        assert "**Core**" in md
        assert "**Tests**" in md
        assert "- **Added** `a.py` — New helper" in md
        assert "- **Modified** `b.py` — Wire it up" in md

    def test_changes_section_caps_file_list(self):
        result = WalkthroughResult(
            summary="Changes.",
            file_changes=[
                WalkthroughFileEntry(f"f{i}.py", FileChangeType.MODIFIED, "x", "Core")
                for i in range(40)
            ],
        )
        md = result.to_markdown()
        assert "<summary><b>What changed</b> — 40 files</summary>" in md
        assert "- **Modified** `f14.py` — x" in md
        assert "- **Modified** `f15.py` — x" not in md
        assert "_…and 25 more files_" in md

    def test_linked_issues_rendered(self):
        from mira.models import LinkedIssue

        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(
            linked_issues=[
                LinkedIssue(
                    identifier="ENG-123",
                    title="Add retry",
                    state="In Progress",
                    url="https://linear.app/acme/issue/ENG-123",
                )
            ]
        )
        assert "### Linked issues" in md
        assert "[ENG-123](https://linear.app/acme/issue/ENG-123) — Add retry *(In Progress)*" in md

    def test_no_linked_issues_section_by_default(self):
        result = WalkthroughResult(summary="Changes.")
        assert "### Linked issues" not in result.to_markdown()

    def test_effort_rendered(self):
        result = WalkthroughResult(
            summary="Changes.",
            effort=WalkthroughEffort(level=4, label="Complex", minutes=45),
        )
        md = result.to_markdown()
        assert "Estimated review effort:" in md
        assert "Complex (4/5)" in md
        assert "~45 min" in md

    def test_diffstat_rendered(self):
        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(reviewed_files=3, additions=120, deletions=30)
        assert "`+120 \u221230`" in md

    def test_effort_suppressed_in_progress(self):
        result = WalkthroughResult(
            summary="Changes.",
            effort=WalkthroughEffort(level=4, label="Complex", minutes=45),
        )
        assert "Estimated review effort:" not in result.to_markdown(in_progress=True)

    def test_require_issue_warns_when_missing(self):
        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(require_issue=True)
        assert "### Linked issues" in md
        assert "No linked issue found" in md

    def test_ticket_criteria_checklist_rendered(self):
        from mira.models import TicketCriterion

        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(
            ticket_criteria=[
                TicketCriterion("ENG-482", "Retries are capped", "met"),
                TicketCriterion("ENG-482", "Failures surface", "unmet", "not handled"),
                TicketCriterion("ENG-482", "Metrics emitted", "unclear"),
            ]
        )
        assert "### Ticket acceptance criteria" in md
        assert "**ENG-482**" in md
        assert "\u2705 Retries are capped" in md
        assert "\u274c Failures surface — not handled" in md
        assert "\u26a0\ufe0f Metrics emitted" in md

    def test_unmet_criterion_forces_request_changes(self):
        from mira.models import TicketCriterion

        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(
            ticket_criteria=[TicketCriterion("ENG-482", "Failures surface", "unmet")]
        )
        assert "## Verdict: \U0001f6d1 Request changes" in md
        assert "**Ticket requirements not met:**" in md
        assert "`ENG-482` — Failures surface" in md

    def test_met_criteria_do_not_change_verdict(self):
        from mira.models import TicketCriterion

        result = WalkthroughResult(summary="Changes.")
        md = result.to_markdown(
            ticket_criteria=[TicketCriterion("ENG-482", "Retries are capped", "met")]
        )
        assert "## Verdict: \u2705 Looks good to merge" in md

    def test_no_confidence_score_no_section(self):
        result = WalkthroughResult(summary="No score.")
        md = result.to_markdown()
        assert "/5" not in md

    def test_help_footer(self):
        result = WalkthroughResult(summary="Footer test.")
        md = result.to_markdown()
        lines = md.split("\n")
        assert "---" in lines, "Expected separator '---' in markdown output"
        separator_idx = len(lines) - 1 - lines[::-1].index("---")
        footer_text = "\n".join(lines[separator_idx:])
        assert "`@miracodeai help`" in footer_text
        assert "available commands and usage tips" in footer_text

    def test_help_footer_custom_bot_name(self):
        result = WalkthroughResult(summary="Footer test.")
        md = result.to_markdown(bot_name="mybot")
        assert "`@mybot help`" in md
        assert "@miracodeai" not in md

    def test_contains_walkthrough_marker(self):
        result = WalkthroughResult(summary="Test.")
        md = result.to_markdown()
        assert md.startswith(WALKTHROUGH_MARKER)
        assert md.count(WALKTHROUGH_MARKER) == 1

    def test_review_stats_params_accepted(self):
        """to_markdown still accepts review_stats/existing_issues params without error."""
        result = WalkthroughResult(summary="Changes.")
        stats = {Severity.BLOCKER: 1, Severity.WARNING: 2}
        md = result.to_markdown(review_stats=stats, existing_issues=3)
        assert "## Mira PR Walkthrough" in md

    def test_clean_output_no_changes_table(self):
        """Walkthrough markdown does not include a changes table."""
        result = WalkthroughResult(
            summary="Changes.",
            file_changes=[
                WalkthroughFileEntry(
                    path="a.py", change_type=FileChangeType.ADDED, description="New file"
                ),
            ],
        )
        md = result.to_markdown()
        assert "### Changes" not in md
        assert "| File |" not in md


class TestDeriveReviewVerdict:
    """The one verdict shared by the walkthrough, review body, and check run."""

    def _blocker(self) -> ReviewComment:
        return ReviewComment(
            path="a.py",
            line=1,
            end_line=None,
            severity=Severity.BLOCKER,
            category="bug",
            title="Boom",
            body="b",
            confidence=0.9,
        )

    def test_clean_review_approves(self):
        assert derive_review_verdict(ReviewResult(summary="ok")).label == "Looks good to merge"

    def test_unmet_criterion_requests_changes(self):
        result = ReviewResult(
            ticket_criteria=[TicketCriterion("EPIC-1", "Do the thing", "unmet", "missing")]
        )
        assert derive_review_verdict(result).label == "Request changes"

    def test_unverified_ticket_downgrades_approval(self):
        result = ReviewResult(
            summary="ok",
            linear_issue_ids=["EPIC-1113"],
            linear_lookup_status="unavailable",
            linear_lookup_detail="no Linear API key set",
        )
        verdict = derive_review_verdict(result)
        assert verdict.label == VERDICT_NEEDS_REVIEW
        assert "EPIC-1113" in ticket_unverified_note(result)
        assert "no Linear API key set" in ticket_unverified_note(result)

    def test_unverified_ticket_keeps_blocking_verdict(self):
        """A blocker already blocks; the ticket note must not soften it."""
        result = ReviewResult(
            comments=[self._blocker()],
            linear_lookup_status="unavailable",
        )
        assert derive_review_verdict(result).label == "Request changes"

    def test_loaded_ticket_is_not_downgraded(self):
        result = ReviewResult(
            summary="ok",
            linear_issue_ids=["EPIC-1113"],
            linear_lookup_status="loaded",
        )
        assert derive_review_verdict(result).label == "Looks good to merge"
        assert ticket_unverified_note(result) == ""


class TestBuildReviewStats:
    def _make_comment(self, severity: Severity) -> ReviewComment:
        return ReviewComment(
            path="f.py",
            line=1,
            end_line=None,
            severity=severity,
            category="test",
            title="t",
            body="b",
            confidence=0.9,
        )

    def test_counts_by_severity(self):
        comments = [
            self._make_comment(Severity.BLOCKER),
            self._make_comment(Severity.BLOCKER),
            self._make_comment(Severity.WARNING),
            self._make_comment(Severity.NITPICK),
        ]
        stats = build_review_stats(comments)
        assert stats == {Severity.BLOCKER: 2, Severity.WARNING: 1, Severity.NITPICK: 1}

    def test_empty_comments(self):
        assert build_review_stats([]) == {}

    def test_single_severity(self):
        comments = [self._make_comment(Severity.SUGGESTION)] * 3
        stats = build_review_stats(comments)
        assert stats == {Severity.SUGGESTION: 3}
