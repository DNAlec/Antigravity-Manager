# Cache affinity phase 1

Scope: the authenticated sub2api AM upstream path and AM's protocol-neutral account affinity and upstream usage diagnostics. Both repositories branch from origin/main. No deployment, billing behavior, prompt cleanup, image retention, or retry policy changes are included.

## Reviewed changes

1. Bridge stable, authenticated-user-scoped root and branch identities from sub2api to AM before protocol conversion. Preserve root affinity and branch isolation using AM's existing composite session convention. Protect the generated identity from static competing header overrides. Content-only fallback remains best effort when clients provide no conversation identity; a cache-affinity hint alone is not a conversation identity.
2. Replace unconditional AM binding writes with selection-time version leases and atomic compare-and-update. Preserve legitimate failover, first-selection convergence, same-account retries, account invalidation, and image scheduling. Late responses and QoS callbacks cannot undo a newer binding.
3. Observe raw upstream usage before protocol conversion or scaling. Distinguish absent cache usage from explicit zero. Add bounded, per-attempt diagnostics without modifying token billing or downstream responses. sub2api's existing upstream recording can compare the AM response with the independently captured Antigravity usage.

## Validation

Run focused session bridge, binding race/ABA/failover, and usage-observation tests. Include cross-protocol bridge coverage, user/key isolation, append-only history, conflicting header overrides, delayed stale success/failure, clearing/deleting accounts, cumulative SSE usage, explicit zero, missing fields, and truncated streams. Do not run release pre-flight or the full application suite.

## Follow-up work

History-prefix stability, image compaction boundaries, model-specific affinity exceptions, retry wait policy, multi-instance binding persistence, and separating sub2api billed usage in its reporting schema remain separate changes.

## Reading the diagnostics

When AM request monitoring is enabled, JSON response logs include a local `_usage_diagnostics` object (also retained in simple payload mode). Non-JSON response logs retain their existing contents and emit the same bounded counters through structured tracing with the request ID. The diagnostic object never appears in the response sent to sub2api and does not change billing fields.

Each observed generation response has its own attempt, model, account ID, source, final observed numeric `raw_usage`, and `transport_end` (`eof`, `error`, `dropped`, or `pending`). Repeated SSE usage updates replace only fields present in that update, including explicit zero; they are not added together. Missing cache counters stay missing. Malformed data, oversized events, invalid counters, and omitted attempts are reported separately. Collection is bounded to 32 attempts, 256 KiB per SSE event, and 128 characters per model/account/source label.

Coverage includes successful HTTP generation responses for the four protocols, image generation/edit, and stream auto-heal continuations. HTTP failures before body observation, countTokens, and background summary requests are outside this instrumentation. `eof` describes transport completion, not semantic answer validity; a stream dropped by a protocol adapter or cancelled by a client is not labeled complete. Do not treat missing or incomplete observations as zero cache hits.

For comparable requests with valid complete Gemini counters, measure token-weighted cache reuse as `sum(cachedContentTokenCount) / sum(promptTokenCount)` within the same model and workload. Use one cumulative observation per attempt, split auto-heal/retry attempts from ordinary responses, and report counter coverage alongside the ratio. Compare this with sub2api's received usage separately from its billing view; ForceCacheBilling and AM token scaling are not raw upstream cache measurements.

## Deployment assumptions and limits

- Keep `proxy.scheduling.affinity_scope` set to `session` (the default), or set `ANTIGRAVITY_SCHEDULING_AFFINITY_SCOPE=session`. API-key affinity would group users behind sub2api's shared AM credential. `PerformanceFirst` deliberately does not create sticky leases. This change does not alter the selected scheduling mode or retry/wait policy.
- The bridge applies only to authenticated AM-upstream requests. Root affinity and branch thinking-history isolation use AM's existing `:agent:` convention. Account failover and model conversion do not change the bridge identity.
- Content fallback cannot distinguish identical anonymous first messages from the same user/key; removing or rewriting the first message can change that fallback. Explicit conversation IDs are preferable. Prompt cache keys alone are not treated as conversation IDs.
- Independent sub2api deployments sharing one AM credential must also avoid colliding user/key ID namespaces. Replicas sharing the same database use the same namespace.
- The new bridge changes existing session/storage keys once, so an initial cold-cache/thinking-store rebuild period is expected. AM bindings remain process-local; restart and cross-instance routing are separate work.
- No production traffic was replayed and no cache-hit uplift has been measured. A controlled rollout should compare warmed, equivalent workloads using raw counters. Production deployment and live cache measurement are separate from this implementation.

## Review notes

Independent review found and corrected a selection-time race: an old request could publish a new initial binding after session clearing or account deletion. Initial claims now validate both the captured session revision and account epoch under the binding lock. Only requests from the same vacant revision can adopt their concurrent initial winner; migration and unbinding terminate that eligibility. Late commits and QoS callbacks cannot overwrite or remove a newer binding.

## Local verification (2026-11-11)

- sub2api: 11 new `TestAMSession*` groups plus existing `TestAMForwardProtocolMatrix` and `TestAMAccountScopeAndAddress` passed with `go test -tags unit ./internal/service -run '^TestAM(Session|ForwardProtocolMatrix|AccountScopeAndAddress)' -count=1`.
- AM: 36 focused tests passed: 10 session-binding registry tests, 4 token-manager lease tests, 8 existing sticky-session tests, 5 image/scheduler tests, 7 raw-usage tests, 1 bridge identity contract test, and 1 debug-stream compatibility test.
- AM test targets compiled using installed Rust 1.99.1, locked offline dependencies, and `--no-default-features --lib`. Disabling Tauri's frontend-embedding feature avoids requiring a `dist/` build for these backend tests. System Rust 1.93.1 is below an existing dependency's minimum version. The build emits existing project warnings; no dependencies were changed.
- Both worktree diffs pass `git diff --check`. Independent peer review closed the stale-claim finding and accepted the raw-usage integration. No release pre-flight, full suite, frontend build, deployment, or live upstream cache experiment was performed.

AM and sub2api each use branch `codex/cache-affinity-phase1`, based on their fetched `origin/main` (`4fe50b4c` and `91d9ca61c`, respectively). Implementation and verification used isolated worktrees. The unrelated sub2api console work remains separate from this change.
