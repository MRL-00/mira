"""Code-grounded replies to developers in review threads (``run_thread_reply``)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig
from mira.models import PRInfo
from mira.platforms.handlers import _file_diff_for, run_thread_reply

_DIFF = """\
diff --git a/src/handler.py b/src/handler.py
--- a/src/handler.py
+++ b/src/handler.py
@@ -10,2 +10,3 @@ def handle(event):
     policy.ensure_create(event)
+    customer = to_customer(event)
     return customer
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # x
+more
"""


def _pr_info(**overrides: Any) -> PRInfo:
    base: dict[str, Any] = {
        "title": "",
        "description": "",
        "base_branch": "",
        "head_branch": "",
        "url": "https://github.com/o/r/pull/7",
        "number": 7,
        "owner": "o",
        "repo": "r",
    }
    base.update(overrides)
    return PRInfo(**base)


def _provider() -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info = AsyncMock(
        return_value=_pr_info(title="Create customers", description="Why", head_sha="abc123")
    )
    provider.get_pr_diff = AsyncMock(return_value=_DIFF)
    provider.get_repo_tree = AsyncMock(return_value=["src/handler.py", "src/policy.py"])
    provider.get_file_content = AsyncMock(
        return_value="class Policy:\n    def ensure_create(self, e):\n        if not e.id: raise\n"
    )
    provider.get_thread_id_for_comment = AsyncMock(return_value="THREAD_1")
    return provider


def _tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _llm(hops: list[dict]) -> MagicMock:
    """An LLM whose ``complete_agentic`` returns the given assistant messages in order."""
    llm = MagicMock()
    llm.complete_agentic = AsyncMock(side_effect=hops)
    llm.complete_with_tools = AsyncMock(return_value="{}")
    return llm


class TestFileDiffFor:
    def test_extracts_only_the_named_file(self):
        out = _file_diff_for(_DIFF, "src/handler.py")
        assert "to_customer" in out
        assert "more" not in out

    def test_missing_file_or_empty_diff(self):
        assert _file_diff_for(_DIFF, "nope.py") == ""
        assert _file_diff_for("", "src/handler.py") == ""


@pytest.mark.asyncio
class TestRunThreadReply:
    async def _run(self, provider: AsyncMock, llm: MagicMock, reply: str) -> None:
        with (
            patch("mira.platforms.handlers.load_config", return_value=MiraConfig()),
            patch("mira.platforms.handlers.create_llm", return_value=llm),
            patch("mira.platforms.handlers._open_store") as store,
        ):
            store.return_value = MagicMock()
            await run_thread_reply(
                provider,
                _pr_info(),
                reply,
                comment_id=99,
                original_suggestion="`to_customer` is called before `ensure_create` validates.",
                comment_node_id="NODE",
                comment_path="src/handler.py",
                comment_line=11,
                actor="alice",
            )

    async def test_model_reads_code_before_answering_a_question(self):
        provider = _provider()
        llm = _llm(
            [
                {"tool_calls": [_tool_call("read_file", {"path": "src/policy.py"})]},
                {
                    "tool_calls": [
                        _tool_call(
                            "submit_thread_reply",
                            {
                                "intent": "question",
                                "reply": "`Policy.ensure_create` raises on a missing id "
                                "(`src/policy.py:3`), so the create path never runs.",
                            },
                            "c2",
                        )
                    ]
                },
            ]
        )

        await self._run(provider, llm, "@mira why is this a problem?")

        # The file the model asked for was fetched at the PR head.
        provider.get_file_content.assert_awaited_once()
        assert provider.get_file_content.await_args.args[1] == "src/policy.py"
        assert provider.get_file_content.await_args.args[2] == "abc123"
        # The second hop received the tool result.
        second_convo = llm.complete_agentic.await_args_list[1].args[0]
        assert any(
            m.get("role") == "tool" and "ensure_create" in m["content"] for m in second_convo
        )
        # The answer was posted in-thread; a question never resolves the thread.
        provider.reply_to_review_comment.assert_awaited_once()
        assert "src/policy.py:3" in provider.reply_to_review_comment.await_args.args[2]
        provider.resolve_threads.assert_not_awaited()

    async def test_prompt_carries_pr_context_and_the_file_diff(self):
        provider = _provider()
        llm = _llm(
            [
                {
                    "tool_calls": [
                        _tool_call(
                            "submit_thread_reply", {"intent": "agreement", "reply": "Thanks."}
                        )
                    ]
                }
            ]
        )

        await self._run(provider, llm, "fair enough")

        prompt = llm.complete_agentic.await_args_list[0].args[0][0]["content"]
        assert "Create customers" in prompt
        assert "`src/handler.py` line 11" in prompt
        assert "+    customer = to_customer(event)" in prompt
        assert "# x" not in prompt  # other files' hunks are not inlined
        assert "is called before `ensure_create`" in prompt
        assert "read_file(path)" in prompt

    async def test_verified_disagreement_resolves_and_records_rejection(self):
        provider = _provider()
        llm = _llm(
            [
                {
                    "tool_calls": [
                        _tool_call(
                            "submit_thread_reply",
                            {
                                "intent": "disagreement",
                                "verified": True,
                                "reply": "You're right — `ensure_create` validates first.",
                            },
                        )
                    ]
                }
            ]
        )
        with (
            patch("mira.platforms.handlers.load_config", return_value=MiraConfig()),
            patch("mira.platforms.handlers.create_llm", return_value=llm),
            patch("mira.platforms.handlers._open_store") as open_store,
        ):
            store = MagicMock()
            open_store.return_value = store
            await run_thread_reply(
                provider,
                _pr_info(),
                "no, ensure_create already validates",
                comment_id=99,
                comment_node_id="NODE",
                comment_path="src/handler.py",
                comment_line=11,
                actor="alice",
            )

        provider.resolve_threads.assert_awaited_once()
        assert provider.resolve_threads.await_args.args[1] == ["THREAD_1"]
        assert store.record_feedback.call_args.kwargs["signal"] == "rejected"
        assert store.record_feedback.call_args.kwargs["actor"] == "alice"

    async def test_unverified_disagreement_leaves_thread_open(self):
        """ "That's handled elsewhere" that the code does not back up cannot clear a finding."""
        provider = _provider()
        llm = _llm(
            [
                {
                    "tool_calls": [
                        _tool_call(
                            "submit_thread_reply",
                            {
                                "intent": "disagreement",
                                "verified": False,
                                "reply": "I checked `src/policy.py` — `ensure_create` only "
                                "checks the id, it does not validate the email.",
                            },
                        )
                    ]
                }
            ]
        )

        await self._run(provider, llm, "no, ensure_create validates the email")

        provider.reply_to_review_comment.assert_awaited_once()
        provider.resolve_threads.assert_not_awaited()

    async def test_falls_back_to_forced_submit_when_hops_run_out(self):
        provider = _provider()
        # Five hops of reading, never submitting.
        llm = _llm([{"tool_calls": [_tool_call("read_file", {"path": "src/policy.py"})]}] * 5)
        llm.complete_with_tools = AsyncMock(
            return_value=json.dumps({"intent": "question", "reply": "It raises on a missing id."})
        )

        await self._run(provider, llm, "why?")

        assert llm.complete_agentic.await_count == 5
        llm.complete_with_tools.assert_awaited_once()
        provider.reply_to_review_comment.assert_awaited_once()

    async def test_no_head_ref_uses_single_call_without_tools(self):
        provider = _provider()
        provider.get_pr_info = AsyncMock(return_value=_pr_info(title="t"))  # no head_sha/branch
        llm = _llm([])
        llm.complete_with_tools = AsyncMock(
            return_value=json.dumps({"intent": "other", "reply": "Noted."})
        )

        await self._run(provider, llm, "hmm")

        llm.complete_agentic.assert_not_awaited()
        llm.complete_with_tools.assert_awaited_once()
        provider.reply_to_review_comment.assert_awaited_once()

    async def test_llm_failure_posts_nothing(self):
        provider = _provider()
        llm = _llm([RuntimeError("boom")])

        await self._run(provider, llm, "why?")

        provider.reply_to_review_comment.assert_not_awaited()
        provider.resolve_threads.assert_not_awaited()
