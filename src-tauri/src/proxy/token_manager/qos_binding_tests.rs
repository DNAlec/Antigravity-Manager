use super::TokenManager;
use crate::proxy::qos_guardian::AccountQoSState;
use crate::proxy::session_bindings::SessionLease;
use crate::proxy::sticky_config::{SchedulingMode, StickySessionConfig};

const SESSION: &str = "qos-warm-session";
const ACCOUNT: &str = "qos-account";

fn warm_session(manager: &TokenManager) -> SessionLease {
    let lease = manager
        .session_accounts
        .claim(&manager.session_accounts.snapshot(SESSION), ACCOUNT, 0)
        .unwrap();
    manager.session_accounts.commit(&lease, || true).unwrap()
}

fn report_slow_response(manager: &TokenManager, lease: &SessionLease) {
    manager.record_qos_ttft(
        ACCOUNT,
        Some(SESSION),
        "PRO",
        50_000.0,
        false,
        Some(100),
        Some(lease),
    );
}

fn prepare_penalty(manager: &TokenManager, lease: &SessionLease, half_open: bool) {
    manager.qos_guardian.set_auto_jail_enabled(true);
    if half_open {
        manager.qos_guardian.states.insert(
            ACCOUNT.into(),
            AccountQoSState::HalfOpen {
                level: 0,
                consecutive_healthy: 0,
                entered_at: std::time::Instant::now(),
            },
        );
    } else {
        report_slow_response(manager, lease);
        assert!(manager.qos_guardian.is_suspicious(ACCOUNT));
    }
}

#[tokio::test]
async fn qos_penalties_preserve_affinity_when_configuration_is_write_locked() {
    for mode in [SchedulingMode::CacheFirst, SchedulingMode::Balance] {
        for half_open in [false, true] {
            for write_locked in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let manager = TokenManager::new(dir.path().to_path_buf());
                manager
                    .update_sticky_config(StickySessionConfig {
                        mode,
                        ..Default::default()
                    })
                    .await;
                let lease = warm_session(&manager);
                prepare_penalty(&manager, &lease, half_open);

                // Reproduce update_sticky_config's critical section while only
                // changing the wait limit; the scheduling mode stays unchanged.
                let writer = if write_locked {
                    let mut config = manager.sticky_config.write().await;
                    config.max_wait_seconds += 1;
                    Some(config)
                } else {
                    None
                };
                report_slow_response(&manager, &lease);
                drop(writer);

                assert!(manager.qos_guardian.is_jailed_or_probing(ACCOUNT));
                assert_eq!(manager.sticky_config.read().await.mode, mode);
                let preserve = write_locked || mode == SchedulingMode::CacheFirst;
                assert_eq!(
                    manager.session_accounts.committed_lease(SESSION).is_some(),
                    preserve,
                    "mode={mode:?}, half_open={half_open}, write_locked={write_locked}"
                );
                assert_eq!(
                    manager.session_accounts.commit(&lease, || true).is_some(),
                    preserve,
                    "preserving affinity must also preserve the in-flight response lease"
                );
            }
        }
    }
}

#[tokio::test]
async fn balance_qos_penalties_cannot_clear_a_newer_session_binding() {
    for half_open in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let manager = TokenManager::new(dir.path().to_path_buf());
        manager
            .update_sticky_config(StickySessionConfig {
                mode: SchedulingMode::Balance,
                ..Default::default()
            })
            .await;
        let stale_lease = warm_session(&manager);
        prepare_penalty(&manager, &stale_lease, half_open);
        let successor = manager
            .session_accounts
            .candidate(&manager.session_accounts.snapshot(SESSION), "newer-account");
        let successor = manager
            .session_accounts
            .commit(&successor, || true)
            .unwrap();

        report_slow_response(&manager, &stale_lease);

        assert!(manager.qos_guardian.is_jailed_or_probing(ACCOUNT));
        assert_eq!(
            manager
                .session_accounts
                .committed_lease(SESSION)
                .unwrap()
                .account_id(),
            "newer-account"
        );
        assert!(manager
            .session_accounts
            .commit(&stale_lease, || true)
            .is_none());
        assert!(manager
            .session_accounts
            .commit(&successor, || true)
            .is_some());
    }
}

#[tokio::test]
async fn image_transient_failure_cools_account_without_evicting_protected_text_sessions() {
    use crate::proxy::session_bindings::scheduling_session_id;

    for mode in [SchedulingMode::CacheFirst, SchedulingMode::Balance] {
        for write_locked in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let manager = TokenManager::new(dir.path().to_path_buf());
            manager
                .update_sticky_config(StickySessionConfig {
                    mode,
                    ..Default::default()
                })
                .await;
            let image_key = scheduling_session_id("image_gen", Some(SESSION)).unwrap();
            let keys = [image_key.as_ref(), SESSION, "other-text-session"];
            let leases = keys.map(|key| {
                let lease = manager
                    .session_accounts
                    .claim(&manager.session_accounts.snapshot(key), ACCOUNT, 0)
                    .unwrap();
                manager.session_accounts.commit(&lease, || true).unwrap()
            });

            // The image handler reports non-terminal connection failures by
            // account ID. Multiple text sessions can be warm on the same account.
            let writer = if write_locked {
                Some(manager.sticky_config.write().await)
            } else {
                None
            };
            manager.mark_account_transient_failure(ACCOUNT, 30, "upstream connection refused");
            drop(writer);

            assert!(manager.rate_limit_tracker.is_rate_limited(ACCOUNT, None));
            let preserve = write_locked || mode == SchedulingMode::CacheFirst;
            for (key, lease) in keys.iter().zip(&leases) {
                assert_eq!(
                    manager.session_accounts.committed_lease(key).is_some(),
                    preserve,
                    "mode={mode:?}, write_locked={write_locked}, key={key}"
                );
                assert_eq!(
                    manager.session_accounts.commit(lease, || true).is_some(),
                    preserve
                );
            }
        }
    }
}
