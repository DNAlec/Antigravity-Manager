//! A single-answer adapter selects one candidate identity for the entire turn.

use serde_json::Value;
use std::sync::{Arc, Mutex};

/// Shared by the response renderer and its error observer. Only a complete
/// decoded response frame pins the identity; the bounded observer may miss a
/// large media frame and must not select a later, unrelated candidate instead.
#[derive(Clone, Default)]
pub struct CandidateSelection {
    index: Arc<Mutex<Option<u64>>>,
}

impl CandidateSelection {
    pub fn index(&self) -> Option<u64> {
        *self.index.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub fn project(&self, value: &mut Value) {
        let raw = if value.get("response").is_some_and(Value::is_object) {
            &mut value["response"]
        } else {
            value
        };
        let Some(candidates) = raw.get_mut("candidates").and_then(Value::as_array_mut) else {
            return;
        };
        let Some(indices) = crate::proxy::mappers::gemini::candidate_indices(candidates) else {
            // Ambiguous indices cannot safely contribute to a single answer.
            candidates.clear();
            return;
        };
        let mut selected = self.index.lock().unwrap_or_else(|error| error.into_inner());
        if selected.is_none() {
            *selected = indices.first().copied();
        }
        let position = indices.iter().position(|index| Some(*index) == *selected);
        if let Some(position) = position {
            let candidate = candidates.remove(position);
            candidates.clear();
            candidates.push(candidate);
        } else {
            candidates.clear();
        }
    }
}
