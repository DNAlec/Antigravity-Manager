//! An HTTP success publishes an account selection, but is not proof that a
//! structured response passed request validation. Confirmation follows the body.

use super::TokenManager;
use crate::proxy::pipeline::stream_error::{
    has_measured_usage, upstream_error_from_event, UpstreamStreamErrorObserver,
};
use crate::proxy::session_bindings::SessionLease;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::sync::Arc;

pub struct SessionResponseConfirmation {
    manager: Arc<TokenManager>,
    lease: Option<SessionLease>,
    observer: Option<UpstreamStreamErrorObserver>,
    rejected_before_output: bool,
}

impl SessionResponseConfirmation {
    pub(super) fn new(manager: Arc<TokenManager>, lease: SessionLease) -> Self {
        Self {
            manager,
            lease: Some(lease),
            observer: None,
            rejected_before_output: false,
        }
    }

    /// JSON failures have no stream observer. An error-only validation refusal
    /// cannot make a previously cold binding warm; any candidate remains
    /// conservative evidence that the upstream may already have prefilled it.
    pub fn observe_json(&mut self, response: &Value) {
        let inner = response.get("response").unwrap_or(response);
        let has_candidates = inner
            .get("candidates")
            .and_then(Value::as_array)
            .is_some_and(|candidates| !candidates.is_empty());
        self.rejected_before_output = !has_candidates
            && upstream_error_from_event(response).is_some_and(|failure| {
                failure.status == 400 && !has_measured_usage(failure.usage.as_ref())
            });
    }

    fn observe_stream(&mut self) {
        if self
            .observer
            .as_ref()
            .is_some_and(|observer| observer.has_output() || observer.has_measured_usage())
        {
            self.confirm();
        }
    }

    fn confirm(&mut self) {
        let Some(lease) = self.lease.take() else {
            return;
        };
        // Only publish cache warmth here. Account health keeps its existing
        // HTTP-success timing, so dropping an old response cannot clear a newer
        // concurrent request's cooldown.
        self.manager.commit_session_lease(&mut Some(lease.clone()));
        self.manager
            .session_accounts
            .finish_pending_response(&lease);
    }
}

impl Drop for SessionResponseConfirmation {
    fn drop(&mut self) {
        let rejected = self.rejected_before_output
            || self
                .observer
                .as_ref()
                .and_then(|observer| observer.failure())
                .is_some_and(|failure| {
                    failure.status == 400
                        && !failure.after_output
                        && !has_measured_usage(failure.usage.as_ref())
                });
        // EOF without a refusal, transport failure, cancellation and an unread
        // body may all follow upstream prefill. Keep their affinity. A stale
        // generation/account epoch still cannot commit after an explicit abandon.
        if !rejected {
            self.confirm();
        } else if let Some(lease) = self.lease.take() {
            self.manager
                .session_accounts
                .finish_pending_response(&lease);
        }
    }
}

/// Own the confirmation for the entire body lifetime, including true streaming
/// responses whose handler has already returned. Inspect the upstream observer
/// before yielding so protocol conversion cannot hide a validation refusal.
pub fn confirm_session_response_stream<S, E>(
    stream: S,
    observer: UpstreamStreamErrorObserver,
    mut confirmation: Option<SessionResponseConfirmation>,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    if let Some(confirmation) = confirmation.as_mut() {
        confirmation.observer = Some(observer);
    }
    async_stream::stream! {
        // Capture before the first poll: dropping an unread body must also
        // preserve a request the upstream already accepted.
        let mut confirmation = confirmation;
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            if let Some(confirmation) = confirmation.as_mut() {
                confirmation.observe_stream();
            }
            yield item;
        }
    }
}
