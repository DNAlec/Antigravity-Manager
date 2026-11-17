# 429 attempt diagnostics

> Historical fork behavior, superseded by the [beta14 migration](testing/upstream_v5.0.4-beta.14_migration.md).

The shared upstream transport records observed HTTP responses and send failures
before protocol handlers decide whether to retry or return. The monitor attaches
these diagnostics to its local log copy, including pure error responses. This does
not change client responses, retry policy, prompt sanitization or account selection.

For JSON response logs, inspect:

```text
response_body._timing.upstream_attempts.attempts
response_body._timing.upstream_attempts.omitted_attempts
```

Each attempt contains:

- `endpoint`: Daily, Sandbox, Prod or Custom; no URL or hostname.
- `status`: HTTP status, or null for a send failure.
- `response_headers_or_error_s`: elapsed time to response headers or send failure.
  This is not generated-token TTFT or total response-body duration.
- `switch_delay_s`: the existing transport's measured fallback delay, when known.
- `outcome`: http_success, http_error or transport_error. HTTP success does not
  establish that a later streaming body completed successfully.

The original `_timing.ttft_attempts` remains unchanged. The new list survives
handler retries and successful calls following failures. It holds up to 64 observed
outcomes per monitored request and explicitly counts omitted records. Diagnostics
contain no account IDs, credentials, request IDs, error bodies or endpoint URLs.

For non-JSON response bodies or an existing non-object `_timing`, inspect the local
`response_headers["x-am-upstream-attempts"]` field. This metadata is not sent as a
response header to the client. Existing response bodies and error text are retained.
JSON diagnostics also survive concise payload storage through `_timing`.

Absence is not evidence of zero attempts: work rejected before transport, cancelled
before an outcome is observed, unmonitored requests, disabled logging or subsequently
pruned logs may have no entry. Records do not identify account rotations or prove
quota exhaustion versus prompt-related rejection. Existing debug logging is not
required, and logging/storage retention settings continue to apply.

This change supplies missing evidence for investigations such as upstream Issues
[#3969](https://github.com/lbjlaq/Antigravity-Manager/issues/3969) and
[#3975](https://github.com/lbjlaq/Antigravity-Manager/issues/3975); it does not claim
to resolve their underlying 429 triggers.
