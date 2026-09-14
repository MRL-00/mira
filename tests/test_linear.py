"""Tests for Linear issue linking (identifier extraction, client, prompt context)."""

from __future__ import annotations

import pytest

import mira.linear as linear_mod
from mira.config import MiraConfig
from mira.linear import (
    LinearClient,
    extract_issue_identifiers,
    format_issues_context,
    issue_identifiers_for_pr,
    resolve_linked_issues,
)
from mira.models import PRInfo


def _pr(
    title: str = "",
    description: str = "",
    head_branch: str = "",
) -> PRInfo:
    return PRInfo(
        title=title,
        description=description,
        base_branch="main",
        head_branch=head_branch,
        url="https://github.com/acme/repo/pull/1",
        number=1,
        owner="acme",
        repo="repo",
    )


class TestExtractIssueIdentifiers:
    def test_finds_identifiers_in_any_order(self):
        text = "ENG-123 fixes the thing, also see MIR-9"
        assert extract_issue_identifiers(text) == ["ENG-123", "MIR-9"]

    def test_deduplicates(self):
        text = "ENG-1 ENG-1 ENG-1"
        assert extract_issue_identifiers(text) == ["ENG-1"]

    def test_filters_common_technical_tokens(self):
        text = "UTF-8, ISO-8601, CVE-2024-1234, SHA-256 but ENG-42 is real"
        assert extract_issue_identifiers(text) == ["ENG-42"]

    def test_team_key_allowlist(self):
        text = "ENG-1 MIR-2 ABC-3"
        assert extract_issue_identifiers(text, ["MIR"]) == ["MIR-2"]

    def test_empty_text(self):
        assert extract_issue_identifiers("") == []

    def test_ignores_lowercase(self):
        assert extract_issue_identifiers("eng-123") == []


class TestIssueIdentifiersForPR:
    def test_scans_title_description_and_branch(self):
        pr = _pr(
            title="ENG-10: add retry",
            description="Closes MIR-20",
            head_branch="feature/ABC-30-retry",
        )
        assert issue_identifiers_for_pr(pr) == ["ENG-10", "MIR-20", "ABC-30"]


class _FakeResponse:
    def __init__(self, payload: dict, error: bool = False) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise linear_mod.httpx.HTTPError("boom")

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Stand-in for ``httpx.AsyncClient`` driven by identifier → response."""

    def __init__(self, responses: dict[str, _FakeResponse], *args, **kwargs) -> None:
        self._responses = responses

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, json=None, headers=None) -> _FakeResponse:
        identifier = json["variables"]["id"]
        return self._responses.get(identifier, _FakeResponse({"data": {"issue": None}}))


def _issue_payload(identifier: str, title: str = "Add retry") -> dict:
    return {
        "data": {
            "issue": {
                "identifier": identifier,
                "title": title,
                "url": f"https://linear.app/acme/issue/{identifier}",
                "description": "Acceptance: retries are bounded.",
                "state": {"name": "In Progress"},
            }
        }
    }


class TestLinearClient:
    async def test_parses_issue(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            linear_mod.httpx,
            "AsyncClient",
            lambda *a, **k: _FakeAsyncClient({"ENG-1": _FakeResponse(_issue_payload("ENG-1"))}),
        )
        issues = await LinearClient("key").fetch_issues(["ENG-1"])
        assert len(issues) == 1
        assert issues[0].identifier == "ENG-1"
        assert issues[0].title == "Add retry"
        assert issues[0].state == "In Progress"
        assert "linear.app" in issues[0].url

    async def test_skips_unknown_issue(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(linear_mod.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient({}))
        assert await LinearClient("key").fetch_issues(["ENG-404"]) == []

    async def test_http_error_is_swallowed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            linear_mod.httpx,
            "AsyncClient",
            lambda *a, **k: _FakeAsyncClient({"ENG-1": _FakeResponse({}, error=True)}),
        )
        assert await LinearClient("key").fetch_issues(["ENG-1"]) == []


class TestResolveLinkedIssues:
    async def test_no_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MIRA_LINEAR_TOKEN", raising=False)
        assert await resolve_linked_issues(_pr(title="ENG-1"), MiraConfig()) == []

    async def test_disabled_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MIRA_LINEAR_TOKEN", "key")
        config = MiraConfig()
        config.linear.enabled = False
        assert await resolve_linked_issues(_pr(title="ENG-1"), config) == []

    async def test_no_identifier_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MIRA_LINEAR_TOKEN", "key")
        assert await resolve_linked_issues(_pr(title="no ticket here"), MiraConfig()) == []

    async def test_malformed_config_returns_empty(self):
        from unittest.mock import MagicMock

        # A misconfigured/duck-typed config object must not raise.
        assert await resolve_linked_issues(_pr(title="ENG-1"), MagicMock()) == []

    async def test_fetches_referenced_issue(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MIRA_LINEAR_TOKEN", "key")
        monkeypatch.setattr(
            linear_mod.httpx,
            "AsyncClient",
            lambda *a, **k: _FakeAsyncClient({"ENG-7": _FakeResponse(_issue_payload("ENG-7"))}),
        )
        issues = await resolve_linked_issues(_pr(title="ENG-7: do it"), MiraConfig())
        assert [i.identifier for i in issues] == ["ENG-7"]


class TestFormatIssuesContext:
    def test_empty(self):
        assert format_issues_context([]) == ""

    def test_includes_intent(self):
        from mira.models import LinkedIssue

        text = format_issues_context(
            [
                LinkedIssue(
                    identifier="ENG-1",
                    title="Add retry",
                    state="In Progress",
                    url="https://linear.app/x/ENG-1",
                    description="Acceptance: bounded retries.",
                )
            ]
        )
        assert "ENG-1" in text
        assert "Add retry" in text
        assert "In Progress" in text
        assert "Acceptance: bounded retries." in text
