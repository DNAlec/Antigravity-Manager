# Shared image counting resource discovery

The Gemini `:countTokens` and `/countTokens` routes use the same pipeline for
every model name. There is no Argon (or other family) allowlist. This does not
change generation routing, account selection, or the native Claude token
estimator.

## Resource selection

Text-only requests use the resolved generation resource directly. For requests
containing images, including file references and nested tool-result images:

1. Try the resolved generation resource first.
2. Before sending client history, count two synthetic 768×768 RGB PNGs with
   identical pixels and different compression. Both counts must be positive
   integers and equal. A successful HTTP status is insufficient.
3. If the resource fails this check, try candidates obtained using the existing
   generic effort-suffix parser. At most three candidates are considered: the
   original resource and up to two suffix removals, covering compound names such
   as `future-model-tiered-xhigh`. An unknown suffix is preserved. No candidate
   is accepted merely because its name looks like a base model.
4. Send the complete client request to the selected resource on the same
   endpoint and account proxy. The envelope builder retains contents, system
   instructions, tools, media and generation configuration, unwraps an existing
   `request`, sets `request.model`, and removes `safetySettings` as required by
   the count endpoint contract.
5. If no candidate passes on the configured endpoints, return an explicit
   error. Never substitute another model family, discard images, or return a
   fixed image estimate as an upstream count.

New model families with the same effort syntax need no model-name patch. A new
upstream contract or a genuinely different naming syntax may still require
implementation changes; resource discovery cannot create upstream support.

## Bounds, caching and diagnostics

- Capability cache keys include account ID, endpoint and exact candidate name.
  Passing results expire after ten minutes; unsupported results after one
  minute. Authentication, rate-limit and transport errors are not cached as
  unsupported capabilities. Proxy reconfiguration clears the cache.
- At most 1,024 capability entries are retained. Concurrent requests for the
  same cached key share one probe, and at most two probe sequences run at once
  per upstream client. Saturated caches stop admitting new entries.
- Each probe call has a 15-second limit. The whole operation, including lock
  waits and response-body reads, shares the existing gateway request deadline
  (`ANTIGRAVITY_GATEWAY_REQUEST_BUDGET_SECS`). A cold image request makes at
  most six probe calls per endpoint plus its actual count call. Cache hits need
  only the actual call when a usable resource is already known.
- `X-Upstream-Model` identifies the resource that returned the successful count.
  `X-Count-Tokens-Source: upstream` indicates provenance.
  `X-Count-Tokens-Validation` is `png-compression` for image-bearing requests or
  `response-schema` otherwise. These headers are not returned as a successful
  count when no upstream resource is available.
- Missing, negative, fractional and nonnumeric `totalTokens` are errors, not
  successful zero counts. Wrapped and unwrapped upstream responses are accepted.
- No persistent settings, dependency changes or platform-specific executables
  are required. Desktop and headless deployments use the same implementation.

## Compatibility and validation boundary

This intentionally removes silent local estimation from the Gemini counting
endpoint, including for text-only upstream failures. No account returns 503;
unusable image resources or malformed responses return 502; the whole-operation
deadline returns 504; upstream HTTP rejection statuses are retained. Clients must
handle a failed count request. The existing local estimator remains available
for its other consumers.

Compression invariance rejects the observed byte-counting failure mode. It is
not an official tokenizer accuracy baseline, and one synthetic PNG pair cannot
prove correctness for every image size, JPEG, file URI, tool schema, audio or
video. Client data is not used for capability probes. Probes call countTokens
only; they never generate model output.

The focused regression tests cover future model names, preservation of valid
original resources, byte-counting rejection, identical decoded probe pixels,
account/endpoint isolation, shared concurrent probes, expiry, timeouts, rate
limits, nested requests and malformed responses. Production acceptance still
requires repeating the real model matrix and checking the client-facing path.

Rollback consists of reverting this change as a unit; there is no data migration
or cache file to remove. The old behavior restores the Argon-only mapping and
silent estimate fallbacks, including their documented limitations.
