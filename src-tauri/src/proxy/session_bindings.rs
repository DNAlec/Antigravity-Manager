//! Versioned sticky bindings. A selection owns a lease, never a session name alone.
use std::collections::HashMap;
use std::sync::Mutex;

pub(crate) const SESSION_BINDING_CHANGED: &str =
    "Session binding changed during token selection; retry request.";

pub(crate) fn is_session_binding_changed_error(error: &str) -> bool {
    error == SESSION_BINDING_CHANGED
}

/// Image queue spillover must never replace the text conversation's binding.
/// Scope after SessionScope resolution, so session and API-key affinities both
/// receive the same separation, including callers outside the image endpoint.
pub(crate) fn scheduling_session_id<'a>(
    quota_group: &str,
    session_id: Option<&'a str>,
) -> Option<std::borrow::Cow<'a, str>> {
    let session_id = session_id.filter(|id| !id.is_empty())?;
    Some(if quota_group == "image_gen" {
        std::borrow::Cow::Owned(format!("image_gen:{session_id}"))
    } else {
        std::borrow::Cow::Borrowed(session_id)
    })
}

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

impl SessionSnapshot {
    pub(crate) fn matches_lease(&self, lease: &SessionLease) -> bool {
        self.session_id == lease.session_id && self.revision == lease.revision
    }
}

struct Binding {
    account_id: Option<String>,
    revision: u64,
    // A provisional first selection is shared with concurrent requests, but it
    // does not have a warm upstream cache until a response confirms success.
    committed: bool,
    pending_responses: usize,
    // Concurrent requests may adopt a first winner from the same vacant snapshot.
    claim_source_revision: Option<u64>,
    // Account removal permits the displaced request to recover only while this
    // exact vacant successor remains unclaimed. Manual clear grants no successor.
    recovery_source_revision: Option<u64>,
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

pub(crate) enum ColdSessionRelease {
    Released(SessionLease),
    Warm,
    Stale,
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
                    committed: false,
                    pending_responses: 0,
                    claim_source_revision: None,
                    recovery_source_revision: None,
                },
            );
        }
        SessionSnapshot {
            session_id: session_id.to_owned(),
            revision: state.sessions[session_id].revision,
        }
    }

    /// A retry keeps its request's generation; it must never borrow a newer
    /// request's warm binding just because it is selecting another account.
    pub fn retry_snapshot(
        &self,
        session_id: &str,
        lease: &SessionLease,
    ) -> Option<SessionSnapshot> {
        if session_id != lease.session_id {
            return None;
        }
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let binding = state.sessions.get(session_id)?;
        if binding.revision != lease.revision
            && !(binding.account_id.is_none()
                && binding.recovery_source_revision == Some(lease.revision)
                && state
                    .account_epochs
                    .get(&lease.account_id)
                    .copied()
                    .unwrap_or(0)
                    != lease.account_epoch)
        {
            return None;
        }
        Some(SessionSnapshot {
            session_id: session_id.to_owned(),
            revision: binding.revision,
        })
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
        self.claim_inner(snapshot, proposed, expected_account_epoch, true)
    }

    /// Retries may not use the initial-selection cohort exception: adopting its
    /// newer winner would grant an old request authority over a concurrent turn.
    pub fn claim_for_retry(
        &self,
        snapshot: &SessionSnapshot,
        proposed: &str,
        expected_account_epoch: u64,
    ) -> Option<SessionLease> {
        self.claim_inner(snapshot, proposed, expected_account_epoch, false)
    }

    fn claim_inner(
        &self,
        snapshot: &SessionSnapshot,
        proposed: &str,
        expected_account_epoch: u64,
        allow_initial_cohort: bool,
    ) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.account_epochs.get(proposed).copied().unwrap_or(0) != expected_account_epoch {
            return None;
        }
        let binding = state.sessions.get(&snapshot.session_id)?;
        if binding.revision != snapshot.revision
            && !(allow_initial_cohort && binding.claim_source_revision == Some(snapshot.revision))
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
                committed: false,
                pending_responses: 0,
                claim_source_revision: Some(snapshot.revision),
                recovery_source_revision: None,
            },
        );
        Some(state.lease(&snapshot.session_id, revision, proposed))
    }

    pub fn bound_lease(&self, session_id: &str) -> Option<SessionLease> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let binding = state.sessions.get(session_id)?;
        Some(state.lease(session_id, binding.revision, binding.account_id.as_deref()?))
    }

    pub fn committed_lease(&self, session_id: &str) -> Option<SessionLease> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let binding = state.sessions.get(session_id)?;
        if !binding.committed {
            return None;
        }
        Some(state.lease(session_id, binding.revision, binding.account_id.as_deref()?))
    }

    /// A dispatched, HTTP-accepted request is not proof of warmth until its
    /// body arrives, but a temporary disk read failure must not preempt it.
    pub fn disk_protected_lease(&self, session_id: &str) -> Option<SessionLease> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let binding = state.sessions.get(session_id)?;
        if !(binding.committed || binding.pending_responses > 0) {
            return None;
        }
        Some(state.lease(session_id, binding.revision, binding.account_id.as_deref()?))
    }

    pub fn finish_pending_response(&self, lease: &SessionLease) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.matches(lease) {
            if let Some(binding) = state.sessions.get_mut(&lease.session_id) {
                if binding.account_id.as_deref() == Some(&lease.account_id) {
                    binding.pending_responses = binding.pending_responses.saturating_sub(1);
                }
            }
        }
    }

    /// Absence from a snapshot is not a reason to revoke a concurrently
    /// accepted response. Check under the same mutation lock as HTTP acceptance.
    pub fn release_unavailable(&self, lease: &SessionLease) -> ColdSessionRelease {
        self.release_with_guard(lease, true, |b| b.committed || b.pending_responses > 0)
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
        self.publish(lease, eligible, true)
    }

    /// HTTP acceptance publishes the account without asserting that its body
    /// passed validation. Existing warm state must survive another accepted turn.
    pub fn accept(
        &self,
        lease: &SessionLease,
        eligible: impl FnOnce() -> bool,
    ) -> Option<SessionLease> {
        self.publish(lease, eligible, false)
    }

    fn publish(
        &self,
        lease: &SessionLease,
        eligible: impl FnOnce() -> bool,
        confirm_success: bool,
    ) -> Option<SessionLease> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.matches(lease) || !eligible() {
            return None;
        }
        let same_account = state.sessions[&lease.session_id].account_id.as_deref()
            == Some(lease.account_id.as_str());
        let committed =
            confirm_success || (same_account && state.sessions[&lease.session_id].committed);
        let pending_responses = if same_account {
            state.sessions[&lease.session_id].pending_responses
        } else {
            0
        } + usize::from(!confirm_success);
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
                committed,
                pending_responses,
                claim_source_revision,
                recovery_source_revision: None,
            },
        );
        Some(state.lease(&lease.session_id, revision, &lease.account_id))
    }

    pub fn abandon(&self, lease: &SessionLease) -> Option<SessionLease> {
        match self.release(lease, false) {
            ColdSessionRelease::Released(next) => Some(next),
            _ => None,
        }
    }

    /// Atomically preserve a concurrently confirmed binding or release a cold
    /// one. Explicit account rejection may still use unconditional abandon.
    pub fn abandon_if_cold(&self, lease: &SessionLease) -> ColdSessionRelease {
        self.release(lease, true)
    }

    fn release(&self, lease: &SessionLease, only_cold: bool) -> ColdSessionRelease {
        self.release_with_guard(lease, only_cold, |b| only_cold && b.committed)
    }

    fn release_with_guard(
        &self,
        lease: &SessionLease,
        allow_vacant: bool,
        preserve: impl FnOnce(&Binding) -> bool,
    ) -> ColdSessionRelease {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.matches(lease) {
            return ColdSessionRelease::Stale;
        }
        match state.sessions[&lease.session_id].account_id.as_deref() {
            Some(account_id) if account_id != lease.account_id => return ColdSessionRelease::Stale,
            // Rotations hold candidates against the vacant generation left by
            // the previous release. They are cold, not stale. Advance that
            // generation before trying another account so late candidates fail CAS.
            None if !allow_vacant => return ColdSessionRelease::Stale,
            _ => {}
        }
        if preserve(&state.sessions[&lease.session_id]) {
            return ColdSessionRelease::Warm;
        }
        let revision = state.revision();
        state.sessions.insert(
            lease.session_id.clone(),
            Binding {
                account_id: None,
                revision,
                committed: false,
                pending_responses: 0,
                claim_source_revision: None,
                recovery_source_revision: None,
            },
        );
        ColdSessionRelease::Released(state.lease(&lease.session_id, revision, &lease.account_id))
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
        let displaced: Vec<_> = state
            .sessions
            .iter()
            .filter(|(_, b)| b.account_id.as_deref() == Some(account_id))
            .map(|(session_id, b)| (session_id.clone(), b.revision))
            .collect();
        for (session_id, source_revision) in displaced {
            let revision = state.revision();
            state.sessions.insert(
                session_id,
                Binding {
                    account_id: None,
                    revision,
                    committed: false,
                    pending_responses: 0,
                    claim_source_revision: None,
                    recovery_source_revision: Some(source_revision),
                },
            );
        }
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
                committed: true,
                pending_responses: 0,
                claim_source_revision: None,
                recovery_source_revision: None,
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
    fn retry_claim_cannot_adopt_a_winner_published_after_snapshot_validation() {
        let bindings = SessionBindings::new();
        let old = claim_current(&bindings, "s", "a");
        let vacant = bindings.abandon(&old).unwrap();
        let retry = bindings.retry_snapshot("s", &vacant).unwrap();
        // Another request claims and confirms B while the retry awaits disk/OAuth.
        let b = bindings.claim(&retry, "b", 0).unwrap();
        bindings.commit(&b, || true).unwrap();
        assert!(bindings.claim_for_retry(&retry, "c", 0).is_none());
        assert!(bindings.retry_snapshot("s", &vacant).is_none());
        assert_eq!(bindings.committed_lease("s").unwrap().account_id(), "b");

        let first = bindings.snapshot("first");
        let a = bindings.claim(&first, "a", 0).unwrap();
        let concurrent = bindings.claim(&first, "b", 0).unwrap();
        assert_eq!(
            a.account_id(),
            concurrent.account_id(),
            "genuine simultaneous first requests still converge on one winner"
        );
    }

    #[test]
    fn retry_snapshot_rejects_migration_but_recovers_unclaimed_account_removal() {
        let bindings = SessionBindings::new();
        let a = claim_current(&bindings, "s", "a");
        let candidate = bindings.candidate(&a.snapshot(), "b");
        assert!(
            bindings.retry_snapshot("s", &candidate).is_some(),
            "a valid rotating candidate need not be the currently bound account"
        );
        assert!(bindings.retry_snapshot("image_gen:s", &a).is_none());
        bindings.invalidate_account("a", || {});
        assert!(bindings.accept(&a, || true).is_none());
        let recovered = bindings.retry_snapshot("s", &a).unwrap();
        let b = bindings.candidate(&recovered, "b");
        let b = bindings.accept(&b, || true).unwrap();
        bindings.commit(&b, || true).unwrap();
        assert!(bindings.retry_snapshot("s", &a).is_none());
        assert!(bindings.retry_snapshot("s", &candidate).is_none());
        bindings.clear();
        assert!(bindings.retry_snapshot("s", &b).is_none());
    }

    #[test]
    fn provisional_selection_is_not_warm_until_success() {
        let bindings = SessionBindings::new();
        let cold = claim_current(&bindings, "s", "a");
        assert!(
            bindings.bound_lease("s").is_some(),
            "concurrent selection shares its first account"
        );
        assert!(bindings.committed_lease("s").is_none());
        assert!(bindings.commit(&cold, || false).is_none());
        assert!(bindings.committed_lease("s").is_none());
        let warm = bindings.commit(&cold, || true).unwrap();
        assert_eq!(bindings.committed_lease("s").unwrap().account_id(), "a");
        let vacant = bindings.abandon(&warm).unwrap();
        let next = bindings.claim(&vacant.snapshot(), "b", 0).unwrap();
        assert!(bindings.committed_lease("s").is_none());
        assert!(
            bindings.commit(&cold, || true).is_none(),
            "late old success cannot warm a new claim"
        );
        assert!(bindings.commit(&next, || true).is_some());
        assert_eq!(bindings.committed_lease("s").unwrap().account_id(), "b");
    }

    #[test]
    fn image_spillover_keeps_text_binding_and_in_flight_lease() {
        for root in ["tenant:sess-root", "tenant:key-tenant"] {
            let bindings = SessionBindings::new();
            let text_key = scheduling_session_id("chat", Some(root)).unwrap();
            let image_key = scheduling_session_id("image_gen", Some(root)).unwrap();
            let text = claim_current(&bindings, &text_key, "warm-a");
            let image = claim_current(&bindings, &image_key, "warm-a");
            // The image queue excludes A while its permits are occupied. Its
            // selection can abandon and migrate only its own scoped binding.
            let vacant_image = bindings.abandon(&image).unwrap();
            let alternative = bindings
                .claim(&vacant_image.snapshot(), "image-b", 0)
                .unwrap();
            assert!(bindings.commit(&alternative, || true).is_some());
            assert_eq!(bindings.get(&text_key).as_deref(), Some("warm-a"));
            assert_eq!(bindings.get(&image_key).as_deref(), Some("image-b"));
            assert!(bindings.commit(&text, || true).is_some());
        }
    }

    #[test]
    fn image_scope_preserves_text_keys_and_absent_affinity() {
        assert!(scheduling_session_id("image_gen", None).is_none());
        assert!(scheduling_session_id("image_gen", Some("")).is_none());
        assert_eq!(
            scheduling_session_id("chat", Some("tenant:sess-a")).as_deref(),
            Some("tenant:sess-a")
        );
        assert_eq!(
            scheduling_session_id("image_gen", Some("tenant:sess-a")),
            scheduling_session_id("image_gen", Some("tenant:sess-a"))
        );
        assert_ne!(
            scheduling_session_id("image_gen", Some("tenant:sess-a")),
            scheduling_session_id("image_gen", Some("tenant:sess-b"))
        );
    }

    #[test]
    fn vacant_cold_candidate_releases_without_weakening_stale_protection() {
        let bindings = SessionBindings::new();
        let first = claim_current(&bindings, "cold", "a");
        let vacancy = bindings.abandon(&first).unwrap();
        let candidate = bindings.candidate(&vacancy.snapshot(), "b");
        let released = match bindings.abandon_if_cold(&candidate) {
            ColdSessionRelease::Released(next) => next,
            _ => panic!("valid candidate on a vacant binding must allow cold recovery"),
        };
        assert!(bindings.commit(&candidate, || true).is_none());
        assert!(bindings.commit(&first, || true).is_none());
        let healthy = bindings.candidate(&released.snapshot(), "c");
        let healthy = bindings.commit(&healthy, || true).unwrap();
        assert!(matches!(
            bindings.abandon_if_cold(&candidate),
            ColdSessionRelease::Stale
        ));
        assert!(matches!(
            bindings.abandon_if_cold(&healthy),
            ColdSessionRelease::Warm
        ));
        let wrong_account = bindings.candidate(&healthy.snapshot(), "b");
        assert!(matches!(
            bindings.abandon_if_cold(&wrong_account),
            ColdSessionRelease::Stale
        ));
        assert_eq!(bindings.get("cold").as_deref(), Some("c"));
        let vacant = bindings.snapshot("epoch");
        let removed = bindings.candidate(&vacant, "b");
        bindings.invalidate_account("b", || {});
        assert!(matches!(
            bindings.abandon_if_cold(&removed),
            ColdSessionRelease::Stale
        ));
    }

    #[test]
    fn cold_release_preserves_confirmation_and_rejects_stale_epochs() {
        let bindings = SessionBindings::new();
        let cold = claim_current(&bindings, "cold", "a");
        let released = match bindings.abandon_if_cold(&cold) {
            ColdSessionRelease::Released(next) => next,
            _ => panic!("cold lease must release"),
        };
        assert!(bindings.commit(&cold, || true).is_none());
        let accepted = bindings.accept(&released, || true).unwrap();
        let warm = bindings.commit(&accepted, || true).unwrap();
        assert!(matches!(
            bindings.abandon_if_cold(&accepted),
            ColdSessionRelease::Warm
        ));
        bindings.invalidate_account("a", || {});
        assert!(matches!(
            bindings.abandon_if_cold(&warm),
            ColdSessionRelease::Stale
        ));
        assert!(bindings.commit(&accepted, || true).is_none());
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
