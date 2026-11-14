use serde_json::Value;
use std::collections::HashMap;

/// Deduplicate cumulative Gemini frames without collapsing distinct calls in
/// the same candidate. Missing IDs use their occurrence within each frame.
#[derive(Default)]
pub(crate) struct ToolCallTracker {
    emitted: HashMap<(u64, String), usize>,
    occurrences: HashMap<(u64, String), usize>,
}

impl ToolCallTracker {
    pub(crate) fn begin_frame(&mut self) {
        self.occurrences.clear();
    }

    pub(crate) fn observe(&mut self, candidate: u64, call: &Value) -> Option<String> {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let mut identity = call.clone();
        if id.is_none() {
            if let Some(object) = identity.as_object_mut() {
                object.remove("id");
            }
        }
        identity.sort_all_objects();
        let key = (candidate, identity.to_string());
        let occurrence = if id.is_some() {
            1
        } else {
            let count = self.occurrences.entry(key.clone()).or_default();
            *count += 1;
            *count
        };
        let emitted = self.emitted.entry(key).or_default();
        if occurrence <= *emitted {
            return None;
        }
        *emitted = occurrence;
        Some(
            id.map(str::to_owned)
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
        )
    }
}
