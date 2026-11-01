//! Process-level controls for isolated upstream response-header experiments.
//! Unset overrides preserve the normal timeout and retry policy.

use std::sync::OnceLock;
use std::time::Duration;

const PRIMARY_ENV: &str = "ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS";
const FALLBACK_ENV: &str = "ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS";
const SINGLE_ATTEMPT_ENV: &str = "ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT";

#[derive(Debug, Default)]
pub(crate) struct UpstreamRequestPolicy {
    primary_secs: Option<u64>,
    fallback_secs: Option<u64>,
    pub single_attempt: bool,
}

pub(crate) fn request_policy() -> &'static UpstreamRequestPolicy {
    static POLICY: OnceLock<UpstreamRequestPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        let policy = UpstreamRequestPolicy::from_lookup(|name| std::env::var(name).ok());
        if policy.is_overridden() {
            tracing::info!(
                ?policy,
                "Upstream request policy overrides enabled (restart to change)"
            );
        }
        policy
    })
}

impl UpstreamRequestPolicy {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let parse_timeout = |name, raw: Option<String>| {
            raw.and_then(|raw| match raw.trim().parse::<u64>() {
                Ok(secs @ 5..=600) => Some(secs),
                _ => {
                    tracing::warn!(
                        variable = name,
                        "Ignoring invalid header timeout; expected 5..=600 seconds"
                    );
                    None
                }
            })
        };
        let primary_secs = parse_timeout(PRIMARY_ENV, lookup(PRIMARY_ENV));
        let fallback_secs = parse_timeout(FALLBACK_ENV, lookup(FALLBACK_ENV));
        let single_attempt = match lookup(SINGLE_ATTEMPT_ENV) {
            None => false,
            Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => {
                    tracing::warn!(
                        variable = SINGLE_ATTEMPT_ENV,
                        "Ignoring invalid single-attempt setting; expected true/false or 1/0"
                    );
                    false
                }
            },
        };
        Self {
            primary_secs,
            fallback_secs,
            single_attempt,
        }
    }

    pub fn is_overridden(&self) -> bool {
        self.primary_secs.is_some() || self.fallback_secs.is_some() || self.single_attempt
    }

    pub fn header_budget(&self, method: &str, body_len: usize) -> HeaderBudget {
        let (primary, fallback) = if method == "streamGenerateContent" {
            (
                self.primary_secs
                    .unwrap_or(((body_len as u64) / (64 * 1024) + 30).clamp(30, 45)),
                self.fallback_secs.unwrap_or(25),
            )
        } else {
            (((body_len as u64) / (256 * 1024) + 15).clamp(15, 30), 15)
        };
        HeaderBudget {
            primary: Duration::from_secs(primary),
            fallback: Duration::from_secs(if self.single_attempt { 0 } else { fallback }),
        }
    }

    pub fn account_attempt_limit(&self, normal_limit: usize) -> usize {
        if self.single_attempt {
            1
        } else {
            normal_limit
        }
    }

    pub fn next_rotation_attempt(
        &self,
        used_attempts: &mut usize,
        max_attempts: usize,
        retry_same_account: bool,
    ) -> Option<usize> {
        // Grace retries must also consume the experimental single-attempt limit.
        if retry_same_account && !self.single_attempt {
            return used_attempts.checked_sub(1);
        }
        if *used_attempts >= self.account_attempt_limit(max_attempts) {
            return None;
        }
        let attempt = *used_attempts;
        *used_attempts += 1;
        Some(attempt)
    }
}

#[derive(Debug)]
pub(crate) struct HeaderBudget {
    primary: Duration,
    fallback: Duration,
}

impl HeaderBudget {
    pub fn total(&self) -> Duration {
        self.primary + self.fallback
    }

    pub fn endpoint_timeout(
        &self,
        endpoint_index: usize,
        remaining: Duration,
        capacity_exhausted: bool,
    ) -> Duration {
        let target = if endpoint_index == 0 {
            self.primary
        } else {
            self.fallback
        };
        let target = if capacity_exhausted {
            target.min(Duration::from_secs(15))
        } else {
            target
        };
        // Never extend a nearly exhausted chain with the old five-second floor.
        target.min(remaining)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(values: &[(&str, &str)]) -> UpstreamRequestPolicy {
        UpstreamRequestPolicy::from_lookup(|name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    #[test]
    fn defaults_preserve_small_and_large_request_budgets() {
        let policy = UpstreamRequestPolicy::default();
        for (method, body_len, primary, fallback) in [
            ("streamGenerateContent", 2016, 30, 25),
            ("streamGenerateContent", 2 * 1024 * 1024, 45, 25),
            ("generateContent", 2016, 15, 15),
        ] {
            let budget = policy.header_budget(method, body_len);
            assert_eq!(
                (budget.primary.as_secs(), budget.fallback.as_secs()),
                (primary, fallback)
            );
        }
    }

    #[test]
    fn longer_header_wait_increases_the_chain_budget_for_small_requests() {
        let policy = policy(&[(PRIMARY_ENV, "180"), (FALLBACK_ENV, "60")]);
        let budget = policy.header_budget("streamGenerateContent", 2016);
        assert_eq!(budget.total(), Duration::from_secs(240));
        assert_eq!(
            budget.endpoint_timeout(0, budget.total(), false),
            Duration::from_secs(180)
        );
        assert_eq!(
            budget.endpoint_timeout(1, Duration::from_secs(60), false),
            Duration::from_secs(60)
        );
        assert_eq!(
            policy.header_budget("countTokens", 2016).total(),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn timeout_never_exceeds_remaining_budget_even_below_five_seconds() {
        let budget = policy(&[(PRIMARY_ENV, "180")]).header_budget("streamGenerateContent", 2016);
        for remaining in [
            Duration::ZERO,
            Duration::from_millis(500),
            Duration::from_secs(2),
        ] {
            assert_eq!(budget.endpoint_timeout(1, remaining, false), remaining);
        }
        assert_eq!(
            budget.endpoint_timeout(1, budget.total(), true),
            Duration::from_secs(15)
        );
    }

    #[test]
    fn invalid_overrides_fall_back_to_bounded_defaults() {
        for invalid in [
            "",
            "0",
            "-1",
            "4",
            "601",
            "1800",
            "abc",
            "18446744073709551616",
        ] {
            let policy = policy(&[(PRIMARY_ENV, invalid), (FALLBACK_ENV, invalid)]);
            assert_eq!(
                policy.header_budget("streamGenerateContent", 2016).total(),
                Duration::from_secs(55)
            );
        }
    }

    #[test]
    fn single_attempt_blocks_rotation_grace_and_endpoint_fallback() {
        let policy = policy(&[(PRIMARY_ENV, "180"), (SINGLE_ATTEMPT_ENV, "true")]);
        let mut used = 0;
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, false), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, true), None);
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, false), None);
        let budget = policy.header_budget("streamGenerateContent", 2016);
        assert_eq!(budget.total(), Duration::from_secs(180));
        assert_eq!(
            budget.endpoint_timeout(1, budget.total(), false),
            Duration::ZERO
        );
    }

    #[test]
    fn normal_policy_keeps_same_account_grace_retry() {
        let policy = UpstreamRequestPolicy::default();
        let mut used = 0;
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, false), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, true), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, false), None);
    }
}
