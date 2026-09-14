"""Tests for the linked-ticket acceptance-criteria verification pass."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from mira.core.ticket_verify import verify_ticket_criteria
from mira.models import LinkedIssue, TicketCriterion


def _llm(raw: str) -> MagicMock:
    llm = MagicMock()
    llm.complete_with_tools = AsyncMock(return_value=raw)
    return llm


def _issues() -> list[LinkedIssue]:
    return [
        LinkedIssue(
            identifier="ENG-482",
            title="Bound payment retries",
            state="In Progress",
            criteria=["Retries are capped", "Terminal failures surface"],
        )
    ]


class TestVerifyTicketCriteria:
    async def test_no_issues_returns_empty(self):
        assert await verify_ticket_criteria(_llm("{}"), [], "diff") == []

    async def test_empty_diff_returns_empty(self):
        assert await verify_ticket_criteria(_llm("{}"), _issues(), "") == []

    async def test_grades_criteria(self):
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "issue": "ENG-482",
                        "criterion": "Retries are capped",
                        "status": "unmet",
                        "evidence": "src/payments.py:42 uses `while True`",
                    },
                    {
                        "issue": "ENG-482",
                        "criterion": "Terminal failures surface",
                        "status": "met",
                        "evidence": "PaymentError raised at src/payments.py:50",
                    },
                ]
            }
        )
        result = await verify_ticket_criteria(_llm(raw), _issues(), "diff --git ...")
        assert [(c.criterion, c.status) for c in result] == [
            ("Retries are capped", "unmet"),
            ("Terminal failures surface", "met"),
        ]
        assert result[0].is_unmet
        assert "src/payments.py:42" in result[0].evidence

    async def test_invalid_status_becomes_unclear(self):
        raw = json.dumps(
            {
                "criteria": [
                    {"issue": "ENG-482", "criterion": "Retries are capped", "status": "maybe"}
                ]
            }
        )
        result = await verify_ticket_criteria(_llm(raw), _issues(), "diff")
        assert result[0].status == "unclear"

    async def test_omitted_explicit_criterion_is_unclear(self):
        """A criterion the model skipped is surfaced, never assumed met."""
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "issue": "ENG-482",
                        "criterion": "Retries are capped",
                        "status": "met",
                        "evidence": "ok",
                    }
                ]
            }
        )
        result = await verify_ticket_criteria(_llm(raw), _issues(), "diff")
        by_text = {c.criterion: c for c in result}
        assert by_text["Retries are capped"].status == "met"
        assert by_text["Terminal failures surface"].status == "unclear"
        assert "not assessed" in by_text["Terminal failures surface"].evidence

    async def test_unknown_issue_attached_to_first_ticket(self):
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "issue": "WRONG-1",
                        "criterion": "Retries are capped",
                        "status": "unmet",
                        "evidence": "x",
                    }
                ]
            }
        )
        result = await verify_ticket_criteria(_llm(raw), _issues(), "diff")
        assert result[0].issue == "ENG-482"

    async def test_llm_failure_marks_explicit_criteria_unclear(self):
        llm = MagicMock()
        llm.complete_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
        result = await verify_ticket_criteria(llm, _issues(), "diff")
        assert [c.status for c in result] == ["unclear", "unclear"]
        assert all(isinstance(c, TicketCriterion) for c in result)

    async def test_unparseable_response_marks_unclear(self):
        result = await verify_ticket_criteria(_llm("not json"), _issues(), "diff")
        assert [c.status for c in result] == ["unclear", "unclear"]

    async def test_no_explicit_criteria_and_failure_returns_empty(self):
        issues = [LinkedIssue(identifier="ENG-1", description="Do a thing.")]
        result = await verify_ticket_criteria(_llm("not json"), issues, "diff")
        assert result == []

    async def test_grades_derived_requirements_when_no_explicit_criteria(self):
        issues = [LinkedIssue(identifier="ENG-1", description="Do a thing.")]
        raw = json.dumps(
            {
                "criteria": [
                    {
                        "issue": "ENG-1",
                        "criterion": "A thing exists",
                        "status": "unmet",
                        "evidence": "missing",
                    }
                ]
            }
        )
        result = await verify_ticket_criteria(_llm(raw), issues, "diff")
        assert result[0].status == "unmet"
