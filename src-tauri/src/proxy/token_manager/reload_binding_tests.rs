use super::TokenManager;
use crate::proxy::sticky_config::SchedulingMode;
use serde_json::{json, Value};
use std::path::Path;

const MODEL: &str = "gemini-2.5-flash";

fn account(id: &str) -> Value {
    json!({"id":id,"email":format!("{id}@example.invalid"),"disabled":false,
       "token":{"access_token":format!("token-{id}"),"refresh_token":"unit-test","expires_in":3600,"expiry_timestamp":chrono::Utc::now().timestamp()+3600,"project_id":"local-project"},
       "quota":{"models":[{"name":MODEL,"percentage":100}]}})
}

fn write(path: &Path, data: &Value) {
    std::fs::write(path, data.to_string()).unwrap();
}

#[tokio::test]
async fn unreadable_account_defers_warm_selection_and_only_confirmed_removal_invalidates_leases() {
    for end in ["recover", "disable", "delete", "unregister"] {
        let dir = tempfile::tempdir().unwrap();
        let accounts = dir.path().join("accounts");
        std::fs::create_dir(&accounts).unwrap();
        // Legacy file names can differ from the internal, indexed account ID.
        let a = accounts.join("legacy-file.json");
        let b = accounts.join("b.json");
        let index = dir.path().join("accounts.json");
        write(&index, &json!({"accounts":[{"id":"a"},{"id":"b"}]}));
        write(&a, &account("a"));
        write(&b, &account("b"));
        let manager = TokenManager::new(dir.path().into());
        manager.sticky_config.write().await.mode = SchedulingMode::CacheFirst;
        assert_eq!(manager.load_accounts().await.unwrap(), 2);
        let bindings = [("one", "a"), ("two", "a"), ("other", "b")];
        let leases = bindings.map(|(key, id)| {
            let claim = manager
                .session_accounts
                .claim(&manager.session_accounts.snapshot(key), id, 0)
                .unwrap();
            manager.session_accounts.commit(&claim, || true).unwrap()
        });
        manager.reload_account("a").await.unwrap();
        assert_eq!(
            manager
                .session_accounts
                .committed_lease("one")
                .unwrap()
                .account_id(),
            "a"
        );
        std::fs::write(&a, b"{\"id\":").unwrap();
        // Repeated full snapshots of unreadable JSON are not account deletions.
        for _ in 0..2 {
            assert_eq!(manager.load_accounts().await.unwrap(), 1);
        }
        assert!(!manager.tokens.contains_key("a"));
        // A selection that started before publication can still be entering or
        // finishing OAuth. Its missing pool entry must not release the warm lease.
        let refresh_error = manager
            .refresh_account_token_for_selection("a", 300, std::time::Duration::from_secs(1))
            .await
            .err()
            .expect("quarantined accounts cannot refresh");
        assert!(refresh_error.is_temporary());
        let mut in_flight = Some(leases[0].clone());
        assert!(manager
            .release_session_after_refresh_failure(
                &refresh_error,
                SchedulingMode::CacheFirst,
                "chat",
                Some("one"),
                "a",
                &mut in_flight,
            )
            .is_some());
        assert_eq!(
            manager
                .get_token("chat", false, Some("new"), MODEL)
                .await
                .unwrap()
                .3,
            "b"
        );
        let err = manager
            .get_token("chat", false, Some("one"), MODEL)
            .await
            .unwrap_err();
        assert!(
            err.contains("CacheFirst bound account state on disk"),
            "{err}"
        );
        for (key, id) in bindings {
            assert_eq!(
                manager
                    .session_accounts
                    .committed_lease(key)
                    .unwrap()
                    .account_id(),
                id
            );
        }
        match end {
            "recover" => {
                write(&a, &account("a"));
                // No separate reload: the next warm request can verify the account again.
                assert_eq!(
                    manager
                        .get_token("chat", false, Some("one"), MODEL)
                        .await
                        .unwrap()
                        .3,
                    "a"
                );
            }
            "disable" => {
                let mut v = account("a");
                v["disabled"] = json!(true);
                write(&a, &v);
            }
            "delete" => std::fs::remove_file(&a).unwrap(),
            "unregister" => write(&index, &json!({"accounts":[{"id":"b"}]})),
            _ => unreachable!(),
        }
        assert_eq!(
            manager.load_accounts().await.unwrap(),
            if end == "recover" { 2 } else { 1 }
        );
        assert!(manager.unreadable_accounts.is_empty());
        if end != "recover" {
            assert!(!manager.missing_pool_refresh_error("a").is_temporary());
        }
        for ((key, id), lease) in bindings.into_iter().zip(leases) {
            let preserved = id == "b" || end == "recover";
            assert_eq!(
                manager.session_accounts.committed_lease(key).is_some(),
                preserved,
                "{end} {key}"
            );
            assert_eq!(
                manager
                    .session_accounts
                    .commit(&lease, || manager.tokens.contains_key(id))
                    .is_some(),
                preserved
            );
        }
    }
}

#[tokio::test]
async fn account_json_updates_publish_only_complete_files_during_full_scans() {
    let dir = tempfile::tempdir().unwrap();
    let accounts = dir.path().join("accounts");
    std::fs::create_dir(&accounts).unwrap();
    let path = accounts.join("a.json");
    write(&path, &account("a"));
    let manager = TokenManager::new(dir.path().into());
    assert_eq!(manager.load_accounts().await.unwrap(), 1);
    let updates = async {
        for revision in 0..24 {
            super::update_account_json(&path, move |json| json["test_revision"] = json!(revision))
                .await
                .unwrap();
        }
    };
    let scans = async {
        for _ in 0..120 {
            assert_eq!(manager.load_accounts().await.unwrap(), 1);
            tokio::task::yield_now().await;
        }
    };
    tokio::join!(updates, scans);
    let final_data: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(final_data["test_revision"], 23);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o077,
            0
        );
    }
}

#[tokio::test]
async fn accepted_first_turn_survives_quarantine_but_not_confirmed_account_removal() {
    use std::{collections::HashSet, sync::Arc};
    for scenario in [
        "healthy",
        "delete",
        "disable",
        "delete-before-body",
        "disable-before-body",
        "preoutput-400-quarantined",
        "quarantine-before-http",
        "quarantine-before-body",
        "preoutput-400",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let accounts = dir.path().join("accounts");
        std::fs::create_dir(&accounts).unwrap();
        let a = accounts.join("a.json");
        let b = accounts.join("b.json");
        write(&a, &account("a"));
        let manager = Arc::new(TokenManager::new(dir.path().into()));
        manager.sticky_config.write().await.mode = SchedulingMode::CacheFirst;
        assert_eq!(manager.load_accounts().await.unwrap(), 1);
        let (_, _, _, first, _, mut lease) = manager
            .get_token_filtered_with_lease("chat", false, Some("s"), MODEL, &HashSet::new())
            .await
            .unwrap();
        assert_eq!(first, "a");

        let break_snapshot = || {
            match scenario {
                "delete" | "delete-before-body" => std::fs::remove_file(&a).unwrap(),
                "disable" | "disable-before-body" => {
                    let mut disabled = account("a");
                    disabled["disabled"] = json!(true);
                    write(&a, &disabled);
                }
                _ => std::fs::write(&a, "{").unwrap(),
            }
            write(&b, &account("b"));
        };
        if ["quarantine-before-http", "delete", "disable"].contains(&scenario) {
            break_snapshot();
            assert_eq!(manager.load_accounts().await.unwrap(), 1);
        }
        let mut response = manager.accept_session_response(&mut lease);
        if [
            "quarantine-before-body",
            "delete-before-body",
            "disable-before-body",
            "preoutput-400-quarantined",
        ]
        .contains(&scenario)
        {
            break_snapshot();
            assert_eq!(manager.load_accounts().await.unwrap(), 1);
        }
        if let Some(response) = response.as_mut() {
            if scenario.starts_with("preoutput-400") {
                response.observe_json(&json!({"error":{"code":400,"message":"Unknown field bad_parameter","status":"INVALID_ARGUMENT"}}));
            } else {
                response.observe_json(&json!({"candidates":[{"content":{"parts":[{"text":"first answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":32500,"candidatesTokenCount":12}}));
            }
        }
        drop(response);
        let committed = manager
            .session_accounts
            .committed_lease("s")
            .map(|l| l.account_id().to_owned());
        let selected = manager
            .get_token("chat", false, Some("s"), MODEL)
            .await
            .map(|t| t.3);
        if scenario == "healthy" {
            assert_eq!(committed.as_deref(), Some("a"));
            assert_eq!(selected.as_deref(), Ok("a"));
        }
        if scenario.starts_with("preoutput-400") {
            assert!(committed.is_none());
        }
        if scenario.starts_with("delete") || scenario.starts_with("disable") {
            assert!(committed.is_none());
            assert_eq!(selected.as_deref(), Ok("b"));
        }
        if scenario.starts_with("quarantine") {
            assert_eq!(committed.as_deref(), Some("a"));
            assert!(selected.unwrap_err().contains("state on disk"));
            write(&a, &account("a"));
            assert_eq!(
                manager
                    .get_token("chat", false, Some("s"), MODEL)
                    .await
                    .unwrap()
                    .3,
                "a"
            );
        }
    }
}

#[tokio::test]
async fn overlapping_accepted_bodies_protect_quarantined_lease_until_resolved() {
    use std::{collections::HashSet, sync::Arc};
    for scenario in [
        "confirmed_before_followup",
        "accepted_body_pending",
        "body_pending_healthy",
        "body_pending_removed",
        "one_refusal_one_pending",
        "all_refused",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let accounts = dir.path().join("accounts");
        std::fs::create_dir(&accounts).unwrap();
        let a = accounts.join("a.json");
        let b = accounts.join("b.json");
        write(&a, &account("a"));
        let m = Arc::new(TokenManager::new(dir.path().into()));
        m.sticky_config.write().await.mode = SchedulingMode::CacheFirst;
        assert_eq!(m.load_accounts().await.unwrap(), 1);
        let (_, _, _, first, _, mut l) = m
            .get_token_filtered_with_lease("chat", false, Some("same"), MODEL, &HashSet::new())
            .await
            .unwrap();
        assert_eq!(first, "a");
        let mut pending = m.accept_session_response(&mut l).unwrap();
        let response = json!({"candidates":[{"content":{"role":"model","parts":[{"text":"first successful answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":32500}});
        if scenario == "confirmed_before_followup" {
            pending.observe_json(&response);
            drop(pending);
            pending = m.accept_session_response(&mut l).unwrap();
        }
        if scenario == "one_refusal_one_pending" || scenario == "all_refused" {
            let mut sibling = m.accept_session_response(&mut l).unwrap();
            sibling.observe_json(&json!({"error":{"code":400,"status":"INVALID_ARGUMENT","message":"invalid argument"}}));
            drop(sibling);
        }
        if scenario == "all_refused" {
            pending.observe_json(&json!({"error":{"code":400,"status":"INVALID_ARGUMENT","message":"invalid argument"}}));
        }
        let mut pending = Some(pending);
        if scenario == "all_refused" {
            drop(pending.take());
        }
        write(&b, &account("b"));
        if scenario == "body_pending_removed" {
            std::fs::remove_file(&a).unwrap()
        } else if scenario != "body_pending_healthy" {
            std::fs::write(&a, "{").unwrap();
        }
        m.load_accounts().await.unwrap();
        let outcome = m
            .get_token_filtered_with_lease("chat", false, Some("same"), MODEL, &HashSet::new())
            .await;
        let next = match outcome {
            Ok((_, _, _, next, _, mut newer)) => {
                m.commit_session_lease(&mut newer);
                next
            }
            Err(e) => e,
        };
        if let Some(mut p) = pending.take() {
            p.observe_json(&response);
            drop(p);
        }
        let committed = m
            .session_accounts
            .committed_lease("same")
            .map(|l| l.account_id().to_string());

        if [
            "confirmed_before_followup",
            "accepted_body_pending",
            "one_refusal_one_pending",
        ]
        .contains(&scenario)
        {
            assert!(next.contains("state on disk"));
            assert_eq!(committed.as_deref(), Some("a"));
        } else if scenario == "body_pending_healthy" {
            assert_eq!(next, "a");
        } else {
            assert_eq!(next, "b");
            assert_eq!(committed.as_deref(), Some("b"));
        }
    }
}
