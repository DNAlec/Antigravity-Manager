# Documentation index

This folder contains developer-focused documentation (architecture, implementation details, and validation steps).

## Proxy
- [AM / sub2api cache continuity handoff (2026-11-14)](plans/cache-continuity-handoff-20261114.md) — verified fixes, outstanding release blockers, and steps to resume development on another machine.
- [`docs/proxy/auth.md`](proxy/auth.md) — proxy authorization modes, expected client behavior, and implementation pointers.
- [`docs/proxy/accounts.md`](proxy/accounts.md) — account lifecycle in the proxy pool (including auto-disable on `invalid_grant`) and UI behavior.

## z.ai (GLM) integration
- [`docs/zai/implementation.md`](zai/implementation.md) — end-to-end “what’s implemented” and how to validate it.
- [`docs/zai/mcp.md`](zai/mcp.md) — MCP endpoints exposed by the proxy (Search / Reader / Vision) and upstream behavior.
- [`docs/zai/provider.md`](zai/provider.md) — Anthropic-compatible passthrough provider details and dispatch modes.
- [`docs/zai/vision-mcp.md`](zai/vision-mcp.md) — built-in Vision MCP server protocol and tool implementations.
- [`docs/zai/notes.md`](zai/notes.md) — research notes, constraints, and future follow-ups (budget/usage, additional endpoints).

## Agent Integrations
- [`docs/jeikcode_integration.md`](jeikcode_integration.md) — JeikCode integration guide, one-click synchronization, configuration options, and KV-cache optimization.
