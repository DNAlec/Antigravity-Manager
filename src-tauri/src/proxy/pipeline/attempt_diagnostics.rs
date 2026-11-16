//! Request-local, bounded transport diagnostics, independent of protocol and TTFT.
//! Never capture URLs, credentials, account identifiers or upstream error bodies.

use crate::proxy::monitor::{ProxyRequestLog, UpstreamRequestBodyHolder};
use crate::proxy::upstream::client::FallbackAttemptLog;
use serde_json::{json, Value};

const MAX_ATTEMPTS: usize = 64;

#[derive(Default)]
pub(crate) struct AttemptDiagnostics {
    attempts: Vec<Value>,
    omitted_attempts: usize,
}

impl AttemptDiagnostics {
    pub(crate) fn record(&mut self, attempt: &FallbackAttemptLog) {
        if self.attempts.len() >= MAX_ATTEMPTS {
            self.omitted_attempts = self.omitted_attempts.saturating_add(1);
            return;
        }
        let endpoint = match attempt.endpoint_name.as_str() {
            "Daily" => "Daily",
            "Sandbox" => "Sandbox",
            "Prod" => "Prod",
            _ => "Custom",
        };
        self.attempts.push(json!({
            "endpoint": endpoint,
            "status": attempt.status,
            // Time to response headers or send failure, not generated-token TTFT.
            "response_headers_or_error_s": attempt.duration_s,
            "switch_delay_s": attempt.switch_delay_s,
            "outcome": if attempt.status.is_none() { "transport_error" }
                else if attempt.is_effective { "http_success" } else { "http_error" },
        }));
    }

    pub(crate) fn snapshot(&self) -> Option<Value> {
        if self.attempts.is_empty() {
            return None;
        }
        Some(json!({
            "attempts": self.attempts,
            "omitted_attempts": self.omitted_attempts,
        }))
    }
}

/// Record at the shared transport boundary before a handler can retry or return.
/// Keep the existing per-call fallback list and its timing semantics unchanged.
pub(crate) fn record_attempt(attempts: &mut Vec<FallbackAttemptLog>, attempt: FallbackAttemptLog) {
    if let Some(holder) = UpstreamRequestBodyHolder::current() {
        holder.record_attempt_diagnostic(&attempt);
    }
    attempts.push(attempt);
}

/// Enrich only the persisted log copy, including pure errors and text responses.
pub(crate) fn attach_to_log(log: &mut ProxyRequestLog, holder: &UpstreamRequestBodyHolder) {
    let Some(diagnostics) = holder.attempt_diagnostics_snapshot() else {
        return;
    };
    if let Some(mut body) = log
        .response_body
        .as_deref()
        .and_then(|body| serde_json::from_str::<Value>(body).ok())
        .filter(Value::is_object)
    {
        if body.get("_timing").is_none() {
            body["_timing"] = json!({});
        }
        if body["_timing"].is_object() {
            body["_timing"]["upstream_attempts"] = diagnostics.clone();
            if let Ok(encoded) = serde_json::to_string(&body) {
                log.response_body = Some(encoded);
                return;
            }
        }
    }

    // Preserve non-JSON bodies and existing non-object _timing values verbatim.
    // This is local audit metadata, never a header sent to the client.
    let mut headers = log
        .response_headers
        .as_deref()
        .and_then(|headers| serde_json::from_str::<Value>(headers).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    headers["x-am-upstream-attempts"] = diagnostics;
    if let Ok(encoded) = serde_json::to_string(&headers) {
        log.response_headers = Some(encoded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::monitor::CURRENT_UPSTREAM_CAPTURE;

    fn attempt(status: Option<u16>) -> FallbackAttemptLog {
        FallbackAttemptLog {
            endpoint_name: "Daily".into(),
            endpoint_url: "https://user:secret@example.invalid?key=secret".into(),
            status,
            duration_s: 1.25,
            switch_delay_s: None,
            error: "sensitive upstream body".into(),
            is_effective: status == Some(200),
        }
    }

    fn log(body: &str) -> ProxyRequestLog {
        serde_json::from_value(json!({
            "id": "test", "timestamp": 0, "method": "POST", "url": "/test",
            "status": 429, "duration": 1, "response_body": body,
            "error": "original error",
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn retains_failed_calls_before_success_without_changing_legacy_attempts() {
        let holder = UpstreamRequestBodyHolder::new();
        CURRENT_UPSTREAM_CAPTURE
            .scope(holder.clone(), async {
                let mut failed_call = Vec::new();
                record_attempt(&mut failed_call, attempt(Some(429)));
                let mut successful_call = Vec::new();
                record_attempt(&mut successful_call, attempt(Some(200)));
                holder.set_attempts(successful_call);
                assert_eq!(failed_call[0].error, "sensitive upstream body");
            })
            .await;
        assert_eq!(holder.take_attempts().unwrap().len(), 1);
        let diagnostics = holder.attempt_diagnostics_snapshot().unwrap();
        assert_eq!(diagnostics["attempts"][0]["status"], 429);
        assert_eq!(diagnostics["attempts"][1]["status"], 200);
        assert!(!diagnostics.to_string().contains("secret"));
        assert!(!diagnostics.to_string().contains("sensitive"));
    }

    #[test]
    fn transport_errors_are_bounded_and_endpoint_labels_are_allowlisted() {
        let mut diagnostics = AttemptDiagnostics::default();
        let mut entry = attempt(None);
        entry.endpoint_name = "private hostname".into();
        for _ in 0..MAX_ATTEMPTS + 2 {
            diagnostics.record(&entry);
        }
        let snapshot = diagnostics.snapshot().unwrap();
        assert_eq!(snapshot["attempts"].as_array().unwrap().len(), MAX_ATTEMPTS);
        assert_eq!(snapshot["omitted_attempts"], 2);
        assert_eq!(snapshot["attempts"][0]["endpoint"], "Custom");
        assert_eq!(snapshot["attempts"][0]["outcome"], "transport_error");
        assert!(snapshot["attempts"][0]["status"].is_null());
    }

    #[test]
    fn error_attachment_preserves_error_and_existing_timing_in_concise_storage() {
        let holder = UpstreamRequestBodyHolder::new();
        holder.record_attempt_diagnostic(&attempt(Some(429)));
        let mut entry = log(r#"{"error":{"code":429},"_timing":{"total_s":2}}"#);
        attach_to_log(&mut entry, &holder);
        let body: Value = serde_json::from_str(entry.response_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["error"]["code"], 429);
        assert_eq!(body["_timing"]["total_s"], 2);
        assert_eq!(entry.error.as_deref(), Some("original error"));
        let concise = crate::proxy::payload_audit::simplify_payload_json(&body);
        assert_eq!(
            concise["_timing"]["upstream_attempts"]["attempts"][0]["status"],
            429
        );
    }

    #[test]
    fn text_and_missing_attempts_do_not_change_response_bodies() {
        let holder = UpstreamRequestBodyHolder::new();
        let mut entry = log("upstream unavailable");
        attach_to_log(&mut entry, &holder);
        assert!(entry.response_headers.is_none());
        holder.record_attempt_diagnostic(&attempt(None));
        attach_to_log(&mut entry, &holder);
        assert_eq!(entry.response_body.as_deref(), Some("upstream unavailable"));
        let headers: Value =
            serde_json::from_str(entry.response_headers.as_ref().unwrap()).unwrap();
        assert_eq!(
            headers["x-am-upstream-attempts"]["attempts"][0]["outcome"],
            "transport_error"
        );
    }
}
