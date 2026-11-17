# Full upstream beta14 migration

## Baseline and decision

User requested full upstream behavior, with client compatibility handled in
sub2api rather than preserving divergent AM gateway implementations.

- Source: https://github.com/lbjlaq/Antigravity-Manager/releases/tag/v5.0.4-beta.14
- Exact upstream commit: `66bc8ba7b0f276c01f07ace0671fb1ccf32a0b7b`.
- Fork base: `c538127c0a0e626e3877a3180beab825d30342f7` (5.0.10).
- References: upstream #3981, #3982, #3983, PR #3980, PR #3978, PR #3970,
  and the portions of PR #3976 included in the published tag. The tag does not
  include every later commit on PR #3976.

This merge preserves upstream ancestry and replaces fork application code,
including fork-only files, with the released upstream implementation. It is
not a conflict-only merge or selective cherry-pick.

## Deliberate distribution exceptions

Keep the fork's Docker Compose-only workflows, Dockerfiles, compose files,
release documentation/history, and 5.0.10 manifest version pending a separate
release. No new tag or production deployment is part of this migration.

App/UpdateNotification/Settings remain check-only for upstream releases;
keep their check_only translations and remove the Homebrew install command
from the Tauri IPC registry. Tauri updater endpoints/signing configuration
remain disabled. All other application code follows the tag exactly.

## Behavior and compatibility

- Upstream system prompt sideloading defaults to Auto, tool-description
  sanitization to enabled. Custom saved rewrite rules are not automatically
  equivalent to the latest recommended presets; inspect effective settings.
- Upstream prompt rewriting includes session-move -> session_relocate. Clients
  using executable tool names must validate declarations, history and return
  mapping together. Any required alias adaptation belongs in sub2api.
- Fork-specific Responses/custom-tool/media/history behavior, session binding,
  QoS configuration, stream-error handling and failed-attempt diagnostics have
  been removed in favor of upstream behavior. Prior fork regression documents
  are historical, not current guarantees.
- Do not interpret absence of x-am-upstream-attempts as a new diagnostic failure.
- sub2api already has an AM boundary converting requests to Messages/Gemini,
  plus custom-tool adaptation. Its existing TestAM conversion suite passes,
  but this is not an end-to-end beta14 or production validation. No sub2api
  source changes were necessary for that suite; production client behavior
  (tools, multi-turn history, streaming, media, cache) remains unverified.

## Data and rollback

Before deployment, stop writes and take a consistent backup of the complete
AM data directory, including configuration, accounts and SQLite files, using
normal operational backup procedures. Do not delete live caches automatically.

Upstream adds a nullable thinking_records.context_anchor column. Fork
write_revision has a default and explicit-column inserts remain compatible.
However upstream config serialization drops fork-only proxy.qos fields.
Rollback must therefore restore the saved configuration/data as well as the
old image. A code rollback alone does not restore discarded fields.

Local rollback ref: `backup/pre-upstream-beta14`. Revert the merge with mainline
parent 1 for a future code rollback; avoid resetting published shared history.

## Validation (2026-11-17)

- Prompt sanitizer: 94 passed.
- Inbound pipeline: 69 passed.
- OpenAI mappers: 76 passed, 1 failed:
  prompt_log_identity_cleanup_only_changes_system_instructions expects
  Top-level text in systemInstruction; the new sideloading behavior moves
  content. The tested mapper and sanitizer are byte-identical to the upstream
  tag; no local patch or ignored test conceals the failure.
- Proxy DB: 19 passed.
- Quota: 6 passed.
- TypeScript: npx tsc --noEmit passed.
- sub2api: go test ./internal/pkg/apicompat -run TestAM -count=1 passed.
- Peer review found a restored Homebrew installation entry; resolved by
  retaining check-only Settings and removing the install IPC registration.
- git diff --check reports one upstream-inherited blank line at EOF in
  src/utils/modelFamily.ts; it is retained to keep upstream source identical.
- No release-only preflight, model probes, production config changes or
  deployment were performed. These tests cannot establish reduced 429 rates.
