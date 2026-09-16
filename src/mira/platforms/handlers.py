"""Platform-neutral webhook handlers — shared by the GitHub and GitLab
webhook layers. Each takes a provider/auth and operates through the engine;
none is tied to a specific platform's payload shape."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mira.config import load_config
from mira.core.engine import ReviewEngine
from mira.core.review_status import tracker as review_tracker
from mira.dashboard.models_config import llm_config_for
from mira.index.store import IndexStore
from mira.llm import create_llm
from mira.llm.prompts.review import build_conversation_prompt
from mira.llm.tool_schemas import SUBMIT_THREAD_REPLY_TOOL
from mira.llm.utils import strip_code_fences, strip_think_blocks

logger = logging.getLogger(__name__)

_REVIEW_KEYWORDS = {"review", "review this", "review this pr"}

_REJECT_KEYWORDS = {"reject", "dismiss", "resolve", "ignore"}

_REVIEW_REST_KEYWORDS = {"review-rest", "review rest", "rest", "continue"}

_HELP_KEYWORDS = {"help", "?", "commands"}

_THREAD_REPLY_ENV = Environment(
    loader=FileSystemLoader(
        str(Path(__file__).resolve().parents[1] / "llm" / "prompts" / "templates")
    ),
    trim_blocks=True,
    lstrip_blocks=True,
)

_THREAD_REPLY_TEMPLATE = _THREAD_REPLY_ENV.get_template("thread_reply.jinja2")

PAUSE_LABEL = "mira-paused"

# Thread replies: how many read/grep hops the model gets before it must answer,
# and how much of the file's diff is inlined into the prompt.
_THREAD_REPLY_MAX_HOPS = 5
_THREAD_REPLY_DIFF_CHARS = 12_000

_PAUSE_KEYWORDS = {"pause"}

_RESUME_KEYWORDS = {"resume"}


def _open_store(owner: str, repo: str, platform: str = "github") -> IndexStore:
    """Open an IndexStore for the given owner/repo."""
    return IndexStore.open(owner, repo, platform=platform)


def _help_message(bot_name: str) -> str:
    """Markdown help comment listing every command Mira understands."""
    return (
        f"### Mira commands\n\n"
        f"Mention `@{bot_name}` in a PR comment followed by one of these verbs:\n\n"
        f"| Command | What it does |\n"
        f"|---|---|\n"
        f"| `@{bot_name} review` | Re-run the full review on this PR. Useful after force-pushes or when you want a fresh pass. |\n"
        f"| `@{bot_name} review-rest` | Review files that were skipped on the first pass because the PR was too large. Aliases: `rest`, `continue`. |\n"
        f"| `@{bot_name} pause` | Pause Mira on this PR. No more reviews until you resume. Adds a `mira-paused` label. |\n"
        f"| `@{bot_name} resume` | Resume Mira on a paused PR and re-review the latest diff. |\n"
        f"| `@{bot_name} help` | Show this message. Aliases: `?`, `commands`. |\n"
        f"| `@{bot_name} <anything else>` | Ask a free-form question about the PR. Mira will reply inline using the PR diff as context. |\n\n"
        f"On an inline review comment Mira posted, reply with `@{bot_name} reject` "
        f"(aliases: `dismiss`, `resolve`, `ignore`) to mark the thread resolved and "
        f"teach Mira not to make similar suggestions in the future.\n\n"
        f"To skip a PR entirely, include `@{bot_name} ignore` in the PR body.\n\n"
        f"Full docs: https://docs.miracode.ai/commands"
    )


async def run_pr_review(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    is_private: bool,
    bot_name: str,
    platform: str = "github",
    pr_title: str = "",
) -> None:
    """Platform-neutral review core: review a PR/MR and post the result.

    Shared by the GitHub and GitLab webhook handlers — everything here goes
    through the ``provider`` abstraction and the engine, so it's the same for
    every platform.
    """
    repo_full = f"{owner}/{repo}"

    # Atomically claim the slot — avoids stacking redundant runs when
    # two concurrent webhooks arrive. Returns False if already reviewing.
    if not review_tracker.try_start(repo_full, number, pr_title, pr_url):
        logger.info("Review already in progress for %s, skipping", pr_url)
        return

    config = load_config()
    from mira.dashboard.models_config import llm_config_for

    llm = create_llm(llm_config_for("review", config.llm))
    indexing_llm = create_llm(llm_config_for("indexing", config.llm))
    security_llm = create_llm(llm_config_for("security", config.llm))
    ticket_llm = create_llm(llm_config_for("ticket", config.llm))
    engine = ReviewEngine(
        config=config,
        llm=llm,
        provider=provider,
        bot_name=bot_name,
        indexing_llm=indexing_llm,
        security_llm=security_llm,
        ticket_llm=ticket_llm,
    )

    from mira.dashboard.api import _app_db

    # Keep visibility current — the blast-radius filter relies on it to avoid
    # naming private repos in a public repo's review.
    try:
        _app_db.set_repo_visibility(owner, repo, is_private, platform=platform)
    except sqlite3.OperationalError as exc:
        logger.debug("set_repo_visibility failed (ignored): %s", exc)

    repo_record = _app_db.get_repo(owner, repo, platform=platform)
    is_indexed = bool(repo_record and repo_record.status == "ready")

    logger.info("Reviewing %s (indexed=%s)", pr_url, is_indexed)
    try:
        result = await engine.review_pr(pr_url)
        review_tracker.complete(repo_full, number)
    except Exception as exc:
        review_tracker.fail(repo_full, number, str(exc))
        raise

    # The walkthrough comment already carries the "more accurate after indexing"
    # nudge for unindexed repos, so we don't post a separate note here — that
    # would repeat on every push.

    logger.info("Review complete for %s", pr_url)

    from mira.models import Severity, build_review_stats
    from mira.outbound_webhooks import (
        REVIEW_COMPLETED,
        REVIEW_HIGH_SEVERITY,
        dispatch_event,
    )

    stats = build_review_stats(result.comments)
    event_data = {
        "repo": repo_full,
        "pr_url": pr_url,
        "number": number,
        "comments": len(result.comments),
        "key_issues": len(result.key_issues),
        "severities": {sev.name.lower(): n for sev, n in stats.items()},
    }
    await dispatch_event(REVIEW_COMPLETED, event_data)
    if any(sev >= Severity.WARNING for sev in stats):
        await dispatch_event(REVIEW_HIGH_SEVERITY, event_data)


async def _react(
    provider: Any,
    pr_url: str,
    number: int,
    owner: str,
    repo: str,
    comment_id: int | None,
    reaction: str,
) -> None:
    """React to the command comment, if the platform gave us one. Best-effort."""
    if comment_id is None:
        return
    from mira.models import PRInfo

    pr_info = PRInfo(
        title="",
        description="",
        base_branch="",
        head_branch="",
        url=pr_url,
        number=number,
        owner=owner,
        repo=repo,
    )
    with contextlib.suppress(Exception):
        await provider.react_to_comment(pr_info, comment_id, reaction)


async def run_pr_command(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    question: str,
    actor: str,
    bot_name: str,
    platform: str = "github",
    pr_title: str = "",
    command_comment_id: int | None = None,
) -> None:
    """Platform-neutral handler for an @-mention command on a PR/MR.

    Dispatches help / review / review-rest / free-form Q&A through the provider
    and engine. Shared by the GitHub and GitLab comment handlers.

    ``command_comment_id`` is the comment that issued the command; when given,
    the bot reacts to it (👀 on pickup, 🚀 on completion, 😕 on failure) so a
    re-review that edits the walkthrough in place still visibly happened.
    """
    repo_full = f"{owner}/{repo}"
    config = load_config()
    from mira.dashboard.models_config import llm_config_for

    llm = create_llm(llm_config_for("review", config.llm))
    indexing_llm = create_llm(llm_config_for("indexing", config.llm))
    security_llm = create_llm(llm_config_for("security", config.llm))
    ticket_llm = create_llm(llm_config_for("ticket", config.llm))

    normalized = question.lower().strip()
    is_review = normalized in _REVIEW_KEYWORDS
    is_review_rest = normalized in _REVIEW_REST_KEYWORDS
    is_help = normalized in _HELP_KEYWORDS

    if is_help:
        pr_info_for_help = await provider.get_pr_info(pr_url)
        await provider.post_comment(pr_info_for_help, _help_message(bot_name))
        logger.info("Help requested on %s by @%s", pr_url, actor)
        return

    if is_review_rest:
        from mira.dashboard.api import _app_db

        progress = _app_db.get_pr_review_progress(owner, repo, number, platform=platform)
        if not progress or not progress.skipped_paths:
            pr_info_for_reply = await provider.get_pr_info(pr_url)
            await provider.post_comment(
                pr_info_for_reply,
                f"> @{actor}: nothing left to review — every file in this "
                "PR has already been covered. 🎉",
            )
            return
        engine = ReviewEngine(
            config=config,
            llm=llm,
            provider=provider,
            bot_name=bot_name,
            indexing_llm=indexing_llm,
            security_llm=security_llm,
            ticket_llm=ticket_llm,
        )
        engine._review_only_paths = set(progress.skipped_paths)  # type: ignore[attr-defined]
        if not review_tracker.try_start(repo_full, number, pr_title, pr_url):
            logger.info("Review already in progress for %s, skipping", pr_url)
            return
        logger.info(
            "review-rest on %s by @%s — %d file(s)", pr_url, actor, len(progress.skipped_paths)
        )
        try:
            await engine.review_pr(pr_url)
            review_tracker.complete(repo_full, number)
        except Exception as exc:
            review_tracker.fail(repo_full, number, str(exc))
            raise
    elif is_review:
        engine = ReviewEngine(
            config=config,
            llm=llm,
            provider=provider,
            bot_name=bot_name,
            indexing_llm=indexing_llm,
            security_llm=security_llm,
            ticket_llm=ticket_llm,
        )
        if not review_tracker.try_start(repo_full, number, pr_title, pr_url):
            logger.info("Review already in progress for %s, skipping", pr_url)
            await _react(provider, pr_url, number, owner, repo, command_comment_id, "confused")
            return
        logger.info("Re-review triggered for %s by @%s", pr_url, actor)
        await _react(provider, pr_url, number, owner, repo, command_comment_id, "eyes")
        try:
            await engine.review_pr(pr_url)
            review_tracker.complete(repo_full, number)
        except Exception as exc:
            review_tracker.fail(repo_full, number, str(exc))
            await _react(provider, pr_url, number, owner, repo, command_comment_id, "confused")
            raise
        await _react(provider, pr_url, number, owner, repo, command_comment_id, "rocket")
    else:
        pr_info = await provider.get_pr_info(pr_url)
        diff_text = await provider.get_pr_diff(pr_info)
        messages = build_conversation_prompt(
            question=question,
            diff_text=diff_text,
            pr_title=pr_info.title,
            pr_description=pr_info.description,
        )
        response = await llm.complete(messages, json_mode=False)
        await provider.post_comment(pr_info, f"> @{actor} asked: {question}\n\n{response}")
        logger.info("Replied to comment on %s", pr_url)


def _file_diff_for(diff_text: str, path: str) -> str:
    """The hunks touching ``path`` in the PR diff, or "" when it is not in the diff."""
    if not diff_text or not path:
        return ""
    try:
        from mira.core.diff_parser import parse_diff

        for f in parse_diff(diff_text).files:
            if f.path == path:
                return "\n".join(h.content for h in f.hunks)[:_THREAD_REPLY_DIFF_CHARS]
    except Exception as exc:  # noqa: BLE001 — context only; never block the reply
        logger.debug("Could not extract file diff for %s: %s", path, exc)
    return ""


async def _thread_reply_tools(provider: Any, pr_info: Any) -> object | None:
    """An agentic tool executor over the PR head, or None if the repo can't be read."""
    ref = getattr(pr_info, "head_sha", "") or getattr(pr_info, "head_branch", "")
    if not ref:
        return None
    try:
        from mira.index.context import ProviderSourceFetcher
        from mira.llm.agentic_tools import AgenticToolExecutor

        tree: list[str] = []
        with contextlib.suppress(Exception):
            tree = await provider.get_repo_tree(pr_info, ref)
        return AgenticToolExecutor(
            source_fetcher=ProviderSourceFetcher(provider, pr_info, ref), repo_tree=tree
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Thread reply tools unavailable: %s", exc)
        return None


async def _thread_reply_agentic(llm: Any, prompt: str, executor: object) -> dict:
    """Run the read/grep loop until the model calls ``submit_thread_reply``."""
    from mira.llm.agentic_tools import AGENTIC_TOOLS

    tools = [*AGENTIC_TOOLS, SUBMIT_THREAD_REPLY_TOOL]
    convo: list[dict] = [{"role": "user", "content": prompt}]
    for _hop in range(_THREAD_REPLY_MAX_HOPS):
        msg = await llm.complete_agentic(convo, tools=tools)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        convo.append(
            {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
        )
        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            if name == "submit_thread_reply":
                return args if isinstance(args, dict) else {}
            result = await executor.execute(name, args)  # type: ignore[attr-defined]
            convo.append({"role": "tool", "tool_call_id": call.get("id") or "", "content": result})
    # Out of hops or the model answered in prose — force the terminal call.
    raw = await llm.complete_with_tools(
        messages=convo, tools=[SUBMIT_THREAD_REPLY_TOOL], temperature=0.0
    )
    data = json.loads(strip_think_blocks(strip_code_fences(raw))) if raw else {}
    return data if isinstance(data, dict) else {}


async def run_thread_reply(
    provider: Any,
    pr_info: Any,
    human_reply: str,
    comment_id: int,
    *,
    original_suggestion: str = "",
    thread_id: str | None = None,
    comment_node_id: str | None = None,
    comment_path: str = "",
    comment_line: int = 0,
    actor: str = "",
    bot_name: str = "miracodeai",
    platform: str = "github",
) -> None:
    """Platform-neutral, code-grounded reply to a human in a review thread.

    The reviewer model gets the PR metadata, the diff for the file under
    discussion, and ``read_file`` / ``grep_repo`` over the PR head, so it can
    check the developer's claim or answer their question against the actual
    code instead of paraphrasing its earlier comment.

    ``disagreement`` verified in the code → reply + resolve the thread + record
    a ``rejected`` feedback signal (same learning signal as an explicit reject).
    ``disagreement`` the code does not support → reply with what was found,
    leave the thread open. ``question`` / ``agreement`` / ``other`` → reply,
    leave open.
    """
    config = load_config()
    llm = create_llm(llm_config_for("review", config.llm))

    # Fill in the PR context the webhook payload didn't carry (title, head sha).
    full_pr = pr_info
    diff_text = ""
    if not getattr(pr_info, "title", "") or not getattr(pr_info, "head_sha", ""):
        with contextlib.suppress(Exception):
            full_pr = await provider.get_pr_info(pr_info.url)
    with contextlib.suppress(Exception):
        diff_text = await provider.get_pr_diff(full_pr)

    prompt = _THREAD_REPLY_TEMPLATE.render(
        user_reply=human_reply or "(empty)",
        original_suggestion=original_suggestion,
        pr_title=getattr(full_pr, "title", "") or "",
        pr_description=(getattr(full_pr, "description", "") or "")[:1500],
        comment_path=comment_path or "(unknown file)",
        comment_line=comment_line,
        file_diff=_file_diff_for(diff_text, comment_path),
    )

    try:
        executor = await _thread_reply_tools(provider, full_pr)
        if executor is not None:
            data = await _thread_reply_agentic(llm, prompt, executor)
            calls = getattr(executor, "call_log", [])
            if calls:
                logger.info(
                    "Thread reply on %s read %d file(s)/search(es): %s",
                    pr_info.url,
                    len(calls),
                    ", ".join(f"{c['tool']}({c['arg']})" for c in calls[:6]),
                )
        else:
            raw = await llm.complete_with_tools(
                messages=[{"role": "user", "content": prompt}],
                tools=[SUBMIT_THREAD_REPLY_TOOL],
                temperature=0.0,
            )
            data = json.loads(strip_think_blocks(strip_code_fences(raw))) if raw else {}
    except Exception as exc:
        logger.warning("Thread reply LLM call failed: %s", exc)
        return

    intent = str(data.get("intent", "other")).lower()
    verified = data.get("verified")
    reply_text = str(data.get("reply", "")).strip()
    if not reply_text:
        logger.warning("Thread reply: empty reply (intent=%s). Skipping.", intent)
        return

    try:
        await provider.reply_to_review_comment(pr_info, comment_id, reply_text)
    except Exception as exc:
        logger.warning("Failed to post thread reply: %s", exc)
        return

    # Only a disagreement the model confirmed in the code clears the thread.
    # An unverified one (or a legacy response without the flag) stays open so
    # a blocker can't be waved away with "that's handled elsewhere".
    if intent == "disagreement" and verified is not False:
        try:
            tid = thread_id
            if tid is None and comment_node_id:
                tid = await provider.get_thread_id_for_comment(comment_node_id, pr_info)
            if tid:
                await provider.resolve_threads(pr_info, [tid])
        except Exception as exc:
            logger.warning("Failed to resolve disagreement thread: %s", exc)
        try:
            store = _open_store(pr_info.owner, pr_info.repo, platform)
            try:
                store.record_feedback(
                    pr_number=pr_info.number,
                    pr_url=pr_info.url,
                    comment_path=comment_path,
                    comment_line=comment_line,
                    comment_category="",
                    comment_severity="",
                    comment_title="",
                    signal="rejected",
                    actor=actor,
                )
            finally:
                store.close()
        except Exception as fb_err:
            logger.debug("Failed to record disagreement feedback: %s", fb_err)

    logger.info(
        "Thread reply (%s%s) on %s: %s",
        intent,
        "" if verified is None else f", verified={verified}",
        pr_info.url,
        reply_text[:80],
    )


async def run_pr_merged_learning(
    provider: Any,
    pr_info: Any,
    bot_name: str,
    merged_by: str,
    platform: str = "github",
) -> None:
    """Platform-neutral merge-time learning: record accept/reject + human-review
    signals and synthesize rules. Shared by GitHub and GitLab."""
    from mira.providers.formatting import parse_bot_comment_metadata

    owner, repo, number, pr_url = pr_info.owner, pr_info.repo, pr_info.number, pr_info.url
    store = _open_store(owner, repo, platform)
    accepted = 0
    human_recorded = 0
    deterministic_rules = 0
    llm_rules = 0
    try:
        existing = store.list_feedback(limit=2000)
        if any(
            e.signal in ("accepted", "human_review") and e.pr_number == number for e in existing
        ):
            logger.info("PR %s already processed for merge-time learning", pr_url)
            return
        rejected_locations = {
            (e.comment_path, e.comment_line)
            for e in existing
            if e.signal == "rejected" and e.pr_number == number
        }

        try:
            bot_threads = await provider.get_all_bot_threads(pr_info)
        except Exception as exc:
            logger.warning("Failed to fetch bot threads for %s: %s", pr_url, exc)
            bot_threads = []

        bot_events: list[dict] = []
        for thread in bot_threads:
            if (thread.path, thread.line) in rejected_locations:
                continue
            meta = parse_bot_comment_metadata(thread.body)
            if not meta["category"]:
                continue
            bot_events.append(
                {
                    "pr_number": number,
                    "pr_url": pr_url,
                    "comment_path": thread.path,
                    "comment_line": thread.line,
                    "comment_category": meta["category"],
                    "comment_severity": meta["severity"],
                    "comment_title": meta["title"],
                    "signal": "accepted",
                    "actor": merged_by,
                    "pr_author": pr_info.author,
                }
            )

        try:
            human_comments = await provider.get_human_review_comments(pr_info, bot_name)
        except Exception as exc:
            logger.warning("Failed to fetch human review comments for %s: %s", pr_url, exc)
            human_comments = []

        human_events: list[dict] = []
        for hc in human_comments:
            body = (hc.body or "").strip()
            if not body:
                continue
            human_events.append(
                {
                    "pr_number": number,
                    "pr_url": pr_url,
                    "comment_path": hc.path,
                    "comment_line": hc.line,
                    "comment_category": "human_review",
                    "comment_severity": "",
                    "comment_title": body[:2000],
                    "signal": "human_review",
                    "actor": hc.author,
                    "pr_author": pr_info.author,
                }
            )

        if bot_events:
            store.record_bulk_feedback(bot_events)
            accepted = len(bot_events)
        if human_events:
            store.record_bulk_feedback(human_events)
            human_recorded = len(human_events)

        from mira.analysis.feedback import synthesize_from_human_reviews, synthesize_rules

        deterministic_rules = synthesize_rules(store)

        if human_recorded > 0:
            try:
                config = load_config()
                from mira.dashboard.models_config import llm_config_for

                indexing_llm = create_llm(llm_config_for("indexing", config.llm))
                llm_rules = await synthesize_from_human_reviews(store, indexing_llm)
            except Exception as exc:
                logger.warning("LLM rule synthesis failed for %s: %s", pr_url, exc)
    finally:
        store.close()

    logger.info(
        "PR merged %s: recorded %d accepted + %d human review events; "
        "upserted %d deterministic + %d LLM rules",
        pr_url,
        accepted,
        human_recorded,
        deterministic_rules,
        llm_rules,
    )
