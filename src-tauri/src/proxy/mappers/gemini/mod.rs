// Gemini mapper 模块
// 负责 v1internal 包装/解包

pub mod collector;
pub mod models;
pub mod streaming;
pub mod wrapper; // [NEW]

// No public exports needed here if unused
pub use wrapper::*;

/// Gemini's protobuf default for an omitted candidate index is zero, including
/// when an explicitly indexed sibling arrives first. Position is not identity.
pub(crate) fn candidate_indices(candidates: &[serde_json::Value]) -> Option<Vec<u64>> {
    let mut seen = std::collections::HashSet::new();
    candidates
        .iter()
        .map(|candidate| {
            let index = match candidate.as_object()?.get("index") {
                None => 0,
                Some(serde_json::Value::String(index))
                    if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) =>
                {
                    index.parse().ok()?
                }
                Some(index) => index.as_u64()?,
            };
            seen.insert(index).then_some(index)
        })
        .collect()
}

/// Normalize alternate JSON integer spelling before assigning tool identities
/// or forwarding native Gemini data. Validate every index before mutating any.
pub(crate) fn normalize_candidate_indices(response: &mut serde_json::Value) -> Option<Vec<u64>> {
    let response = if response.get("response").is_some() {
        &mut response["response"]
    } else {
        response
    };
    let Some(candidates) = response
        .get_mut("candidates")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return Some(Vec::new());
    };
    let indices = candidate_indices(candidates)?;
    for (candidate, index) in candidates.iter_mut().zip(&indices) {
        if candidate["index"].is_string() {
            candidate["index"] = serde_json::Value::from(*index);
        }
    }
    Some(indices)
}

/// A synthetic interruption or the protobuf default did not complete a
/// model generation and cannot authorize storing its history.
pub(crate) fn has_candidate_finish(candidate: &serde_json::Value) -> bool {
    candidate["gatewayInterrupted"] != true
        && candidate
            .get("finishReason")
            .or_else(|| candidate.get("finish_reason"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| !s.is_empty() && s != "FINISH_REASON_UNSPECIFIED")
}

/// Both inline media and provider-hosted files are visible Gemini output.
pub(crate) fn has_media(part: &serde_json::Value) -> bool {
    ["inlineData", "inline_data", "fileData", "file_data"]
        .iter()
        .any(|key| part.get(*key).is_some())
}

/// The content required for a deliverable Gemini candidate. Thoughts alone
/// cannot pass the stream lifecycle's no-content guard.
pub(crate) fn has_visible_content(part: &serde_json::Value) -> bool {
    (part.get("thought").and_then(serde_json::Value::as_bool) != Some(true)
        && part.get("text").is_some())
        || part.get("functionCall").is_some()
        || has_media(part)
        || crate::proxy::pipeline::stream_error::has_code_execution_output(part)
}

/// Error/refusal terminals finish transport, but cannot authorize replaying
/// partial reasoning. A model output limit is a valid resumable truncation.
pub(crate) fn has_successful_candidate_finish(candidate: &serde_json::Value) -> bool {
    has_candidate_finish(candidate)
        && matches!(
            candidate
                .get("finishReason")
                .or_else(|| candidate.get("finish_reason"))
                .and_then(serde_json::Value::as_str),
            Some("STOP" | "MAX_TOKENS")
        )
}

#[cfg(test)]
mod candidate_index_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decimal_string_indices_normalize_without_changing_protobuf_defaults() {
        for wrapped in [false, true] {
            let body = json!({"candidates":[{"index":"1","content":{"parts":[{"text":"second"}]}},{"content":{"parts":[{"text":"first"}]}}]});
            let mut response = if wrapped {
                json!({"response":body})
            } else {
                body
            };
            assert_eq!(normalize_candidate_indices(&mut response), Some(vec![1, 0]));
            let body = response.get("response").unwrap_or(&response);
            assert_eq!(body["candidates"][0]["index"], 1);
            assert!(body["candidates"][1].get("index").is_none());
            assert_eq!(
                body["candidates"][0]["content"]["parts"][0]["text"],
                "second"
            );
            let once = response.clone();
            assert_eq!(normalize_candidate_indices(&mut response), Some(vec![1, 0]));
            assert_eq!(response, once);
        }
    }

    #[test]
    fn invalid_and_ambiguous_indices_fail_before_normalization() {
        for candidates in [
            json!([{"index":"0"}, {"index":0}]),
            json!([{"index":"0"}, {}]),
            json!([{}, {}]),
            json!([{"index":"1"}, {"index":"01"}]),
            json!([{"index":"-1"}]),
            json!([{"index":"1.0"}]),
            json!([{"index":" 1"}]),
            json!([{"index":"+1"}]),
            json!([{"index":"18446744073709551616"}]),
            json!([{"index":null}]),
        ] {
            let mut body = json!({"candidates":candidates});
            let original = body.clone();
            assert_eq!(normalize_candidate_indices(&mut body), None, "{original}");
            assert_eq!(body, original);
        }
    }
}
