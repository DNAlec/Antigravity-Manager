//! Process-level single-attempt control for isolated upstream experiments.
//! Timeout handling uses the upstream pipeline's upload and first-byte budgets.

use std::sync::OnceLock;

const SINGLE_ATTEMPT_ENV: &str = "ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT";

#[derive(Debug, Default)]
pub(crate) struct UpstreamRequestPolicy {
    pub single_attempt: bool,
}

pub(crate) fn request_policy() -> &'static UpstreamRequestPolicy {
    static POLICY: OnceLock<UpstreamRequestPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        for name in ["ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS", "ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS"] {
            if std::env::var_os(name).is_some() {
                tracing::warn!(variable = name,
                    "Legacy header override retired; using upstream timeouts. Configure proxy.stream_timeouts.initial_ttft_secs and proxy.upload_timeouts instead");
            }
        }
        UpstreamRequestPolicy::from_lookup(|name| std::env::var(name).ok())
    })
}

impl UpstreamRequestPolicy {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
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
        Self { single_attempt }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_attempt_blocks_rotation_and_grace_retry() {
        let policy = UpstreamRequestPolicy::from_lookup(|_| Some("true".into()));
        let mut used = 0;
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, false), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, true), None);
        assert_eq!(policy.next_rotation_attempt(&mut used, 12, false), None);
        assert_eq!(policy.account_attempt_limit(3), 1);
    }

    #[test]
    fn default_policy_keeps_same_account_grace_retry() {
        let policy = UpstreamRequestPolicy::default();
        let mut used = 0;
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, false), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, true), Some(0));
        assert_eq!(policy.next_rotation_attempt(&mut used, 1, false), None);
    }

    #[test]
    fn invalid_or_disabled_override_keeps_normal_attempts() {
        for value in [None, Some("false"), Some("0"), Some("invalid")] {
            let policy = UpstreamRequestPolicy::from_lookup(|_| value.map(str::to_owned));
            assert_eq!(policy.account_attempt_limit(3), 3);
        }
    }
}
