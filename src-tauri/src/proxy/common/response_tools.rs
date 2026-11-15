use serde_json::Value;
use std::collections::HashMap;

/// Deduplicate cumulative Gemini frames without collapsing distinct calls in
/// the same candidate. Missing IDs use their occurrence within each frame.
#[derive(Default)]
pub(crate) struct ToolCallTracker {
    emitted: HashMap<(u64, String), Vec<TrackedToolCall>>,
    occurrences: HashMap<(u64, String), usize>,
    next_index: usize,
}

#[derive(Clone)]
pub(crate) struct TrackedToolCall {
    pub id: String,
    pub index: usize,
    pub is_new: bool,
}

impl ToolCallTracker {
    pub(crate) fn begin_frame(&mut self) {
        self.occurrences.clear();
    }

    pub(crate) fn observe(&mut self, candidate: u64, call: &Value) -> Option<String> {
        let tracked = self.track_snapshot(candidate, call);
        tracked.is_new.then_some(tracked.id)
    }

    pub(crate) fn track_snapshot(&mut self, candidate: u64, call: &Value) -> TrackedToolCall {
        // Delta emitters still identify a complete argument snapshot. They do
        // not yet merge later argument updates into an already-emitted call.
        self.track_call(candidate, call, false)
    }

    /// Return stable identity even for a repeated frame, so collectors can merge
    /// trailing signatures without emitting or executing the call again.
    pub(crate) fn track(&mut self, candidate: u64, call: &Value) -> TrackedToolCall {
        self.track_call(candidate, call, true)
    }

    fn track_call(
        &mut self,
        candidate: u64,
        call: &Value,
        stable_explicit_id: bool,
    ) -> TrackedToolCall {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let identity = if let Some(id) = id.filter(|_| stable_explicit_id) {
            format!("id:{id}")
        } else {
            let mut identity = call.clone();
            if id.is_none() {
                if let Some(object) = identity.as_object_mut() {
                    object.remove("id");
                }
            }
            identity.sort_all_objects();
            format!("call:{identity}")
        };
        let key = (candidate, identity);
        let occurrence = if id.is_some() {
            1
        } else {
            let count = self.occurrences.entry(key.clone()).or_default();
            *count += 1;
            *count
        };
        let emitted = self.emitted.entry(key).or_default();
        if let Some(previous) = emitted.get(occurrence - 1) {
            return TrackedToolCall {
                is_new: false,
                ..previous.clone()
            };
        }
        let tracked = TrackedToolCall {
            id: id
                .map(str::to_owned)
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
            index: self.next_index,
            is_new: true,
        };
        self.next_index += 1;
        emitted.push(tracked.clone());
        tracked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn r26_complete_call_observer_keeps_existing_argument_update_behavior() {
        let mut tracker = ToolCallTracker::default();
        let first = json!({"id":"same","name":"inspect","args":{"path":"first"}});
        let updated = json!({"id":"same","name":"inspect","args":{"path":"updated"}});
        assert!(tracker.observe(0, &first).is_some());
        tracker.begin_frame();
        assert!(tracker.observe(0, &first).is_none());
        assert!(tracker.observe(0, &updated).is_some());
    }

    #[test]
    fn r26_tool_identity_preserves_occurrences_candidates_and_explicit_ids() {
        let mut tracker = ToolCallTracker::default();
        let anonymous = json!({"name":"inspect","args":{"path":"src"}});
        let first = tracker.track(0, &anonymous);
        let second = tracker.track(0, &anonymous);
        let other_candidate = tracker.track(1, &anonymous);
        assert_ne!(first.id, second.id);
        assert_ne!(first.index, other_candidate.index);
        tracker.begin_frame();
        assert_eq!(tracker.track(0, &anonymous).id, first.id);
        assert_eq!(tracker.track(0, &anonymous).id, second.id);
        assert!(!tracker.track(1, &anonymous).is_new);

        let explicit = json!({"id":"first","name":"inspect","args":{}});
        let distinct = json!({"id":"second","name":"inspect","args":{}});
        assert!(tracker.track(0, &explicit).is_new);
        assert!(tracker.track(0, &distinct).is_new);
        tracker.begin_frame();
        let updated = json!({"id":"first","name":"inspect","args":{"path":"src"}});
        assert!(!tracker.track(0, &updated).is_new);
        assert!(tracker.track(1, &explicit).is_new);
    }
}
