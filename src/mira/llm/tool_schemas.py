"""Structured-output tool schemas the model fills in via tool calling.

The ``submit_*`` schemas for the review, critique, thread-reply, and
walkthrough passes. Distinct from ``agentic_tools`` (the read_file/grep
tools the model calls while reasoning) — these are the shapes it returns.
"""

SUBMIT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit your code review findings including comments, key issues, and a summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "description": "List of review comments on specific lines of code.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path."},
                            "line": {
                                "type": "integer",
                                "description": "Line number in the target file.",
                            },
                            "end_line": {
                                "type": ["integer", "null"],
                                "description": "End line for multi-line comments, or null.",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["blocker", "warning", "suggestion", "nitpick"],
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "bug",
                                    "security",
                                    "performance",
                                    "error-handling",
                                    "race-condition",
                                    "resource-leak",
                                    "maintainability",
                                    "clarity",
                                    "configuration",
                                    "other",
                                ],
                            },
                            "title": {"type": "string", "description": "Short title (<80 chars)."},
                            "body": {
                                "type": "string",
                                "description": "Detailed explanation of the issue. Use single backticks for inline code references. Do NOT use triple-backtick code blocks.",
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "existing_code": {
                                "type": "string",
                                "description": "Verbatim copy of the code from the diff that this comment targets. Must be an exact substring.",
                            },
                            "suggestion": {
                                "type": ["string", "null"],
                                "description": "Optional replacement code to fix the issue. Raw code only — do NOT wrap in backticks or markdown fences.",
                            },
                            "agent_prompt": {
                                "type": ["string", "null"],
                                "description": "Concise imperative instruction for AI coding agents.",
                            },
                        },
                        "required": [
                            "path",
                            "line",
                            "severity",
                            "category",
                            "title",
                            "body",
                            "confidence",
                            "existing_code",
                        ],
                    },
                },
                "key_issues": {
                    "type": "array",
                    "description": "1-3 most critical findings a human reviewer MUST examine.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string"},
                            "path": {"type": "string"},
                            "line": {"type": "integer"},
                        },
                        "required": ["issue", "path", "line"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "Brief overall summary of the review.",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "reviewed_files": {"type": "integer"},
                        "skipped_reason": {"type": ["string", "null"]},
                    },
                },
            },
            "required": ["comments", "summary"],
        },
    },
}

SUBMIT_CRITIQUE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_critique",
        "description": (
            "For each draft review comment, grade how well the evidence in "
            "the diff supports the claimed issue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "description": "One verdict per draft comment, in input order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Zero-based index of the draft comment.",
                            },
                            "evidence": {
                                "type": "string",
                                "enum": ["proven", "plausible", "unsupported"],
                                "description": (
                                    "proven: the shown code demonstrates the issue and the "
                                    "reasoning is correct. "
                                    "plausible: the issue is consistent with the shown code "
                                    "but depends on behaviour or code not shown — this is a "
                                    "valid grade for real findings, not a failure. "
                                    "unsupported: the shown code contradicts the claim, the "
                                    "reasoning is wrong, or it's a style preference dressed "
                                    "up as an issue."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "One short sentence explaining the verdict.",
                            },
                        },
                        "required": ["index", "evidence", "reason"],
                    },
                },
            },
            "required": ["verdicts"],
        },
    },
}


SUBMIT_THREAD_REPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_thread_reply",
        "description": (
            "Reply to a developer's comment on one of your inline PR review "
            "comments, after checking the code. Classify their intent and write "
            "a short, specific reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["disagreement", "question", "agreement", "other"],
                    "description": (
                        "disagreement = developer says the comment was wrong / doesn't apply. "
                        "question = developer asks why it matters or how to fix it. "
                        "agreement = developer acknowledges or thanks. "
                        "other = anything else (off-topic, unclear)."
                    ),
                },
                "verified": {
                    "type": ["boolean", "null"],
                    "description": (
                        "For disagreement only: true when the code confirms the "
                        "developer's point (the thread should be resolved); false when "
                        "the code does not support it and the concern stands. Null for "
                        "other intents."
                    ),
                },
                "reply": {
                    "type": "string",
                    "description": (
                        "Your reply, 2-4 sentences, GitHub markdown allowed (backticks, "
                        'path:line). No emojis, no apologies, no "as an AI". Cite the '
                        "code you checked."
                    ),
                },
            },
            "required": ["intent", "reply"],
        },
    },
}


SUBMIT_TICKET_VERIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_ticket_verification",
        "description": (
            "Grade every requirement of the linked tracker ticket against the "
            "pull request diff. Every requirement must appear exactly once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "description": "One entry per requirement, in ticket order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "Ticket identifier, e.g. ENG-482.",
                            },
                            "criterion": {
                                "type": "string",
                                "description": "The requirement, quoted or closely paraphrased.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["met", "unmet", "unclear"],
                                "description": (
                                    "met: the diff demonstrably satisfies this requirement. "
                                    "unmet: the diff itself shows it was done wrong or "
                                    "contradicts the requirement. unclear: the diff does not "
                                    "prove it either way — including when the decisive code is "
                                    "outside the diff, or the requirement is that existing "
                                    "behaviour stays unchanged."
                                ),
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "One short sentence citing the file:line or the missing "
                                    "work that justifies the status."
                                ),
                            },
                        },
                        "required": ["issue", "criterion", "status", "evidence"],
                    },
                },
            },
            "required": ["criteria"],
        },
    },
}


SUBMIT_WALKTHROUGH_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_walkthrough",
        "description": "Submit a high-level walkthrough summary of the pull request.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief overall summary of the PR."},
                "confidence_score": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "label": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "label", "reason"],
                },
                "change_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "change_type": {
                                            "type": "string",
                                            "enum": ["added", "modified", "deleted", "renamed"],
                                        },
                                        "description": {"type": "string"},
                                    },
                                    "required": ["path", "change_type", "description"],
                                },
                            },
                        },
                        "required": ["label", "files"],
                    },
                },
                "effort": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "minimum": 1, "maximum": 5},
                        "label": {"type": "string"},
                        "minutes": {"type": "integer"},
                    },
                    "required": ["level", "label", "minutes"],
                },
                "sequence_diagram": {
                    "type": ["string", "null"],
                    "description": (
                        "A Mermaid `graph LR` flow diagram showing how the changed files "
                        "relate (imports/data flow), or null when the changes are unrelated."
                    ),
                },
            },
            "required": ["summary", "change_groups"],
        },
    },
}
