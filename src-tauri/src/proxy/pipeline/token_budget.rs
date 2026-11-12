//! Account selection keeps a work watchdog while explicit cooldowns consume
//! the request budget instead of being mistaken for a stalled selection.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::time::Instant;

pub(crate) struct TokenAcquisitionBudget {
    work_deadline: Instant,
    request_deadline: Instant,
    cooldown_ms: AtomicU64,
}

impl TokenAcquisitionBudget {
    pub(crate) fn new() -> Self {
        let request_deadline = super::budget::current_deadline()
            .map(Instant::from_std)
            .unwrap_or_else(|| {
                Instant::now()
                    + Duration::from_secs(super::budget::DEFAULT_GATEWAY_REQUEST_BUDGET_SECS)
            });
        Self::with_limits(Duration::from_secs(15), request_deadline)
    }

    fn with_limits(work: Duration, request_deadline: Instant) -> Self {
        Self {
            work_deadline: Instant::now() + work,
            request_deadline,
            cooldown_ms: AtomicU64::new(0),
        }
    }

    fn deadline(&self) -> Instant {
        (self.work_deadline + Duration::from_millis(self.cooldown_ms.load(Ordering::Relaxed)))
            .min(self.request_deadline)
    }

    pub(crate) async fn run<T>(
        &self,
        work: impl Future<Output = Result<T, String>>,
    ) -> Result<T, String> {
        tokio::pin!(work);
        loop {
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(self.deadline()) => {
                    // A cooldown may have extended the work deadline after this
                    // timer was armed. The request deadline never moves.
                    if Instant::now() >= self.request_deadline {
                        return Err("Token acquisition request budget exhausted; retry later.".into());
                    }
                    if Instant::now() >= self.deadline() {
                        return Err("Token acquisition timeout (15s of work) - system too busy or deadlock detected".into());
                    }
                }
                result = &mut work => return result,
            }
        }
    }

    pub(crate) async fn wait_for_cooldown(&self, wait: Duration) -> Result<(), String> {
        // Preserve the existing 30s reserve for the actual upstream request.
        let remaining = self
            .request_deadline
            .saturating_duration_since(Instant::now());
        if wait > remaining.saturating_sub(Duration::from_secs(30)) {
            return Err("Account cooldown exceeds remaining request budget; retry later.".into());
        }
        self.cooldown_ms
            .fetch_add(wait.as_millis() as u64, Ordering::Relaxed);
        tokio::time::sleep(wait).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cooldown_extends_only_the_work_watchdog() {
        let budget = TokenAcquisitionBudget::with_limits(
            Duration::from_millis(30),
            Instant::now() + Duration::from_secs(60),
        );
        assert!(budget
            .run(async {
                budget.wait_for_cooldown(Duration::from_millis(80)).await?;
                Ok(())
            })
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn ordinary_work_still_expires() {
        let budget = TokenAcquisitionBudget::with_limits(
            Duration::from_millis(30),
            Instant::now() + Duration::from_secs(60),
        );
        let error = budget
            .run(std::future::pending::<Result<(), String>>())
            .await
            .unwrap_err();
        assert!(super::super::errors::is_transient_token_error(&error));
    }

    #[tokio::test]
    async fn exhausted_request_budget_is_not_a_rotation_signal() {
        let budget = TokenAcquisitionBudget::with_limits(
            Duration::from_secs(15),
            Instant::now() + Duration::from_millis(30),
        );
        let error = budget
            .run(std::future::pending::<Result<(), String>>())
            .await
            .unwrap_err();
        assert!(!super::super::errors::is_transient_token_error(&error));
        let error = budget
            .wait_for_cooldown(Duration::from_secs(u64::MAX))
            .await
            .unwrap_err();
        assert!(!super::super::errors::is_transient_token_error(&error));
    }
}
