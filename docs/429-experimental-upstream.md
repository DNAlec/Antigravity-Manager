# 429 experimental upstream integration

This change is separate from the confirmed failed-attempt diagnostics fix. It
is not evidence that all production 429 responses have the same cause.

## Sources

- [PR #3976](https://github.com/lbjlaq/Antigravity-Manager/pull/3976),
  cubelikeplayDaniel, head `87b67d3c94a33b452ca08f7dd7ff7050c0b52b6a`:
  OpenCode session tool instruction rewrites, inline code handling, and the
  Claude interactive-agent preamble (issues #3969, #3972, #3975).
- [PR #3980](https://github.com/lbjlaq/Antigravity-Manager/pull/3980),
  Lucas-Qh-Lai, head `5742aebf547662b6a9627a02a762f88fae504459`:
  Responses MCP namespace restoration (issue #3979). This repairs tool-call
  compatibility; it is not independently a demonstrated 429 fix.

Both PRs remained open when checked on 2026-11-17. PR #3977 was closed without
merging for further testing and refinement and is excluded. Upstream beta14
version and release-document changes are excluded; the fork remains 5.0.9.

## Fork adaptations and behavior changes

The original functional patches were applied before resolving fork conflicts.
Responses continues to use the fork's shared tool-call builder, preserving
custom tools and media behavior for streaming, nonstreaming, and WebSocket use.

Preamble stripping is bounded to known sentences; identity normalization does
not discard arbitrary appended instructions. Header cleaning still protects
inline code examples. Known tool directives and user-configured rewrite rules
can now match inline backticks, while fenced code remains protected. Review
existing custom rules before production deployment: broad rules may now affect
inline examples. Existing saved preset lists are not automatically replaced.

Peer review rejected the session-move → session_relocate portion of #3976:
rewriting executable names in instructions while preserving declarations and
historical calls can make clients call nonexistent tools. Both built-in rename
and frontend/backend presets are excluded. A payload regression test keeps
instructions, declarations and historical calls consistent. This portion needs
an end-to-end alias mapping before it can safely be integrated.

## Validation and rollback

Use focused prompt-sanitizer and OpenAI mapper/handler tests plus TypeScript
checking. Production account/egress behavior requires a deployment comparison;
local tests cannot establish a reduction in 429 responses. Revert this entire
experimental commit to restore prior rewriting and namespace behavior while
keeping the independent failed-attempt diagnostics commit.

Validation on 2026-11-17: final prompt sanitizer 89 passed; OpenAI mappers 149 passed
and 2 failed; OpenAI handlers 131 passed and 1 failed; `npx tsc --noEmit`
passed. All three failures reproduced individually on pre-experiment commit
`37e5a3d7` with the same assertions:

- `server_authoritative_ignores_client_reasoning_content_and_budget`
- `test_openai_responses_api_vs_chat_api_thinking_and_signature`
- `failed_empty_and_refused_streams_do_not_save_history`

They remain unresolved baseline limitations, not a fully green test suite.
