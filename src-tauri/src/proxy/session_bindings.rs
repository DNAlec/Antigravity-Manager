//! Versioned sticky bindings. A selection owns a lease, never a session name alone.
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct SessionLease {
    session_id: String,
    revision: u64,
    account_id: String,
    account_epoch: u64,
}

impl SessionLease {
    pub fn account_id(&self) -> &str {
        &self.account_id
    }
    pub(crate) fn set_account_epoch(&mut self, epoch: u64) {
        self.account_epoch = epoch;
    }
    pub(crate) fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            session_id: self.session_id.clone(),
            revision: self.revision,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SessionSnapshot {
    session_id: String,
    revision: u64,
}

struct Binding {
    account_id: Option<String>,
    revision: u64,
    // Concurrent requests may adopt a first winner from the same vacant snapshot.
    claim_source_revision: Option<u64>,
}

#[derive(Default)]
struct State {
    sessions: HashMap<String, Binding>,
    account_epochs: HashMap<String, u64>,
    next_revision: u64,
}

impl State {
    fn revision(&mut self) -> u64 {
        self.next_revision += 1;
        self.next_revision
    }
    fn lease(&self, session_id: &str, revision: u64, account_id: &str) -> SessionLease {
        SessionLease {
            session_id: session_id.to_owned(),
            revision,
            account_id: account_id.to_owned(),
            account_epoch: self.account_epochs.get(account_id).copied().unwrap_or(0),
        }
    }
    fn matches(&self, lease: &SessionLease) -> bool {
        self.sessions
            .get(&lease.session_id)
            .is_some_and(|b| b.revision == lease.revision)
            && self
                .account_epochs
                .get(&lease.account_id)
                .copied()
                .unwrap_or(0)
                == lease.account_epoch
    }
}

#[derive(Default)]
pub(crate) struct SessionBindings {
    state: Mutex<State>,
}

impl SessionBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn account_epochs(&self) -> HashMap<String, u64> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .account_epochs
            .clone()
    }

    pub fn snapshot(&self, session_id: &str) -> SessionSnapshot {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.sessions.contains_key(session_id) {
            let revision = state.revision();
            state.sessions.insert(
                session_id.to_owned(),
                Binding {
                    account_id: None,
                    revision,
                    claim_source_revision: None,
                },
            );
        }
        SessionSnapshot {
            session_id: session_id.to_owned(),
            revision: state.sessions[session_id].revision,
        }
    }

    /// A rotating candidate may commit only against the state observed before selection.
    pub fn candidate(&self, snapshot: &SessionSnapshot, account_id: &str) -> SessionLease {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.lease(&snapshot.session_id, snapshot.revision, account_id)
    }

    /// Publish only against the selection-time session and account generations.
    /// Concurrent first selections may still adopt their cohort's existing winner.
    pub fn claim(
        &self,
        snapshot: &SessionSnapshot,
        proposed: &str,
        expected_account_epoch: u64,
    ) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.account_epochs.get(proposed).copied().unwrap_or(0) != expected_account_epoch {
            return None;
        }
        let binding = state.sessions.get(&snapshot.session_id)?;
        if binding.revision != snapshot.revision
            && binding.claim_source_revision != Some(snapshot.revision)
        {
            return None;
        }
        if let Some(account_id) = binding.account_id.as_deref() {
            return Some(state.lease(&snapshot.session_id, binding.revision, account_id));
        }
        let revision = state.revision();
        state.sessions.insert(
            snapshot.session_id.clone(),
            Binding {
                account_id: Some(proposed.to_owned()),
                revision,
                claim_source_revision: Some(snapshot.revision),
            },
        );
        Some(state.lease(&snapshot.session_id, revision, proposed))
    }

    pub fn bound_lease(&self, session_id: &str) -> Option<SessionLease> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let binding = state.sessions.get(session_id)?;
        Some(state.lease(session_id, binding.revision, binding.account_id.as_deref()?))
    }

    #[cfg(test)]
    pub fn get(&self, session_id: &str) -> Option<String> {
        self.bound_lease(session_id).map(|lease| lease.account_id)
    }

    /// The eligibility check shares the mutation lock with account removal.
    pub fn commit(
        &self,
        lease: &SessionLease,
        eligible: impl FnOnce() -> bool,
    ) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.matches(lease) || !eligible() {
            return None;
        }
        let same_account = state.sessions[&lease.session_id].account_id.as_deref()
            == Some(lease.account_id.as_str());
        let claim_source_revision = if same_account {
            state.sessions[&lease.session_id].claim_source_revision
        } else {
            None
        };
        let revision = if same_account {
            lease.revision
        } else {
            state.revision()
        };
        state.sessions.insert(
            lease.session_id.clone(),
            Binding {
                account_id: Some(lease.account_id.clone()),
                revision,
                claim_source_revision,
            },
        );
        Some(state.lease(&lease.session_id, revision, &lease.account_id))
    }

    pub fn abandon(&self, lease: &SessionLease) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.matches(lease)
            || state.sessions[&lease.session_id].account_id.as_deref()
                != Some(lease.account_id.as_str())
        {
            return None;
        }
        let revision = state.revision();
        state.sessions.insert(
            lease.session_id.clone(),
            Binding {
                account_id: None,
                revision,
                claim_source_revision: None,
            },
        );
        Some(state.lease(&lease.session_id, revision, &lease.account_id))
    }

    pub fn remove(&self, session_id: &str) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sessions
            .remove(session_id)
            .and_then(|b| b.account_id)
    }

    pub fn clear(&self) {
        // Keep the revision counter: a recreated session must not accept an old lease.
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sessions
            .clear();
    }

    pub fn invalidate_account(&self, account_id: &str, remove_account: impl FnOnce()) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *state
            .account_epochs
            .entry(account_id.to_owned())
            .or_default() += 1;
        state
            .sessions
            .retain(|_, b| b.account_id.as_deref() != Some(account_id));
        remove_account();
    }

    #[cfg(test)]
    pub fn insert(&self, session_id: String, account_id: String) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let revision = state.revision();
        state.sessions.insert(
            session_id,
            Binding {
                account_id: Some(account_id),
                revision,
                claim_source_revision: None,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim_current(bindings: &SessionBindings, session: &str, account: &str) -> SessionLease {
        let snapshot = bindings.snapshot(session);
        let epoch = bindings.account_epochs().get(account).copied().unwrap_or(0);
        bindings.claim(&snapshot, account, epoch).unwrap()
    }

    #[test]
    fn migration_rejects_old_success_failure_and_aba() {
        let bindings = SessionBindings::new();
        let old_a = claim_current(&bindings, "s", "a");
        let b = bindings.candidate(&bindings.snapshot("s"), "b");
        let b = bindings.commit(&b, || true).unwrap();
        assert!(bindings.commit(&old_a, || true).is_none());
        assert!(bindings.abandon(&old_a).is_none());
        let new_a = bindings.candidate(&bindings.snapshot("s"), "a");
        bindings.commit(&new_a, || true).unwrap();
        assert!(bindings.abandon(&old_a).is_none());
        assert!(bindings.abandon(&b).is_none());
        assert_eq!(bindings.get("s").as_deref(), Some("a"));
    }

    #[test]
    fn first_claims_converge_and_same_account_success_is_idempotent() {
        let bindings = SessionBindings::new();
        let snapshot = bindings.snapshot("s");
        let a1 = bindings.claim(&snapshot, "a", 0).unwrap();
        // A successful response must not invalidate another initial request's snapshot.
        bindings.commit(&a1, || true).unwrap();
        let a2 = bindings.claim(&snapshot, "b", 0).unwrap();
        assert_eq!(a2.account_id(), "a");
        assert!(bindings.abandon(&a2).is_some());
    }

    #[test]
    fn first_successful_rotation_wins() {
        let bindings = SessionBindings::new();
        claim_current(&bindings, "s", "a");
        let snapshot = bindings.snapshot("s");
        let b = bindings.candidate(&snapshot, "b");
        let c = bindings.candidate(&snapshot, "c");
        let b = bindings.commit(&b, || true).unwrap();
        assert!(bindings.commit(&c, || true).is_none());
        assert!(bindings.abandon(&b).is_some());
    }

    #[test]
    fn clear_recreate_and_account_removal_invalidate_in_flight_candidates() {
        let bindings = SessionBindings::new();
        let old = claim_current(&bindings, "s", "a");
        bindings.clear();
        assert!(bindings.commit(&old, || true).is_none());
        claim_current(&bindings, "s", "a");
        assert!(bindings.abandon(&old).is_none());
        let candidate = bindings.candidate(&bindings.snapshot("s"), "b");
        bindings.invalidate_account("b", || {});
        assert!(bindings.commit(&candidate, || true).is_none());
        assert_eq!(bindings.get("s").as_deref(), Some("a"));
    }

    #[test]
    fn grace_retry_can_reclaim_only_its_own_vacant_generation() {
        let bindings = SessionBindings::new();
        let initial = claim_current(&bindings, "s", "a");
        let retry = bindings.abandon(&initial).unwrap();
        assert!(bindings.commit(&initial, || true).is_none());
        let renewed = bindings.commit(&retry, || true).unwrap();
        let retry = bindings.abandon(&renewed).unwrap();
        claim_current(&bindings, "s", "b");
        assert!(bindings.commit(&retry, || true).is_none());
        assert_eq!(bindings.get("s").as_deref(), Some("b"));
    }

    #[test]
    fn removed_or_ineligible_account_cannot_commit() {
        let bindings = SessionBindings::new();
        let lease = claim_current(&bindings, "s", "a");
        bindings.invalidate_account("a", || {});
        assert!(bindings.commit(&lease, || true).is_none());
        let fresh = claim_current(&bindings, "s", "a");
        assert!(bindings.commit(&fresh, || false).is_none());
    }

    #[test]
    fn clearing_a_selection_snapshot_prevents_provisional_republication() {
        let bindings = SessionBindings::new();
        let old = bindings.snapshot("s");
        bindings.clear();
        assert!(bindings.claim(&old, "a", 0).is_none());
        assert!(bindings.get("s").is_none());
        let fresh = bindings.snapshot("s");
        assert!(bindings.claim(&old, "a", 0).is_none());
        assert!(bindings.get("s").is_none());
        assert!(bindings.claim(&fresh, "b", 0).is_some());
    }

    #[test]
    fn removed_selected_account_cannot_publish_into_a_vacant_session() {
        let bindings = SessionBindings::new();
        let snapshot = bindings.snapshot("s");
        let selected_epoch = bindings.account_epochs().get("a").copied().unwrap_or(0);
        bindings.invalidate_account("a", || {});
        assert!(bindings.claim(&snapshot, "a", selected_epoch).is_none());
        assert!(bindings.get("s").is_none());
        assert!(bindings.claim(&snapshot, "b", 0).is_some());
    }

    #[test]
    fn migration_and_abandonment_end_the_initial_claim_cohort() {
        let bindings = SessionBindings::new();
        let initial = bindings.snapshot("s");
        bindings.claim(&initial, "a", 0).unwrap();
        let b = bindings.candidate(&bindings.snapshot("s"), "b");
        let b = bindings.commit(&b, || true).unwrap();
        assert!(bindings.claim(&initial, "c", 0).is_none());
        assert_eq!(bindings.get("s").as_deref(), Some("b"));
        let before_abandon = bindings.snapshot("s");
        bindings.abandon(&b).unwrap();
        assert!(bindings.claim(&before_abandon, "a", 0).is_none());
        assert!(bindings.get("s").is_none());
    }

    #[test]
    fn concurrent_initial_requests_share_one_binding() {
        let bindings = std::sync::Arc::new(SessionBindings::new());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let tasks: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|account| {
                let bindings = bindings.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let snapshot = bindings.snapshot("s");
                    barrier.wait();
                    bindings.claim(&snapshot, account, 0).unwrap()
                })
            })
            .collect();
        barrier.wait();
        let leases: Vec<_> = tasks.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(leases[0].account_id(), leases[1].account_id());
    }
}
