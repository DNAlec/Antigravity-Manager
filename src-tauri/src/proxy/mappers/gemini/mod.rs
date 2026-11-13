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
    if candidates.len() == 1 {
        return Some(vec![match candidates[0].as_object()?.get("index") {
            None => 0,
            Some(index) => index.as_u64()?,
        }]);
    }
    let mut seen = std::collections::HashSet::new();
    candidates
        .iter()
        .map(|candidate| {
            let index = match candidate.as_object()?.get("index") {
                None => 0,
                Some(index) => index.as_u64()?,
            };
            seen.insert(index).then_some(index)
        })
        .collect()
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
