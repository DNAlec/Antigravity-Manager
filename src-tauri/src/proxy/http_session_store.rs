// HTTP 会话历史存储
// 为 /v1/responses POST 提供 previous_response_id 链式历史支持
// 这样即使客户端用 HTTP 而不是 WebSocket，也能实现多轮对话

use crate::proxy::handlers::openai::get_cached_tool_call;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const SESSION_TTL_SECS: u64 = 3600; // 1小时过期

#[derive(Debug, Clone)]
pub struct HttpSessionEntry {
    /// 对话历史：instructions + 所有 input items（包括历史response输出）
    pub input_items: Vec<Value>,
    /// 系统指令
    pub instructions: String,
    /// 模型名
    pub model: String,
    /// 上次访问时间（用于TTL淘汰）
    pub last_accessed: Instant,
}

#[derive(Debug)]
struct SessionNode {
    parent: Option<Arc<SessionNode>>,
    input_delta: Vec<Value>,
    response_output: Vec<Value>,
    instructions: String,
    model: String,
    routing_session_id: String,
    // Only an identity actually used for this turn's account/session scope.
    // Never promote a generated response-chain cursor to an upstream affinity.
    scope_identity: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionParent(Arc<SessionNode>);

impl SessionParent {
    pub fn routing_session_id(&self) -> &str {
        &self.0.routing_session_id
    }
    pub fn scope_identity(&self) -> Option<&str> {
        self.0.scope_identity.as_deref()
    }
}

struct StoredSession {
    node: Arc<SessionNode>,
    last_accessed: Instant,
}

struct HttpSessionStore {
    sessions: HashMap<String, StoredSession>,
}

impl HttpSessionStore {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    fn get(&mut self, response_id: &str) -> Option<(HttpSessionEntry, SessionParent)> {
        let stored = self.sessions.get_mut(response_id)?;
        stored.last_accessed = Instant::now();
        let node = stored.node.clone();
        Some((
            HttpSessionEntry {
                input_items: materialize_history(&node),
                instructions: node.instructions.clone(),
                model: node.model.clone(),
                last_accessed: stored.last_accessed,
            },
            SessionParent(node),
        ))
    }

    fn insert(&mut self, response_id: String, entry: HttpSessionEntry) {
        self.insert_delta(
            response_id,
            None,
            entry.input_items,
            Vec::new(),
            entry.instructions,
            entry.model,
            None,
        );
    }

    fn insert_delta(
        &mut self,
        response_id: String,
        parent: Option<SessionParent>,
        input_delta: Vec<Value>,
        response_output: Vec<Value>,
        instructions: String,
        model: String,
        routing_session_id: Option<String>,
    ) {
        self.insert_delta_with_identity(
            response_id,
            parent,
            input_delta,
            response_output,
            instructions,
            model,
            routing_session_id,
            None,
        )
    }

    fn insert_delta_with_identity(
        &mut self,
        response_id: String,
        parent: Option<SessionParent>,
        input_delta: Vec<Value>,
        response_output: Vec<Value>,
        instructions: String,
        model: String,
        routing_session_id: Option<String>,
        scope_identity: Option<String>,
    ) {
        let routing_session_id = routing_session_id.unwrap_or_else(|| response_id.clone());
        self.sessions.insert(
            response_id,
            StoredSession {
                node: Arc::new(SessionNode {
                    parent: parent.map(|parent| parent.0),
                    input_delta,
                    response_output,
                    instructions,
                    model,
                    routing_session_id,
                    scope_identity,
                }),
                last_accessed: Instant::now(),
            },
        );
        // 顺便淘汰过期 session（惰性清理）
        self.evict_expired();
    }

    fn evict_expired(&mut self) {
        let ttl = Duration::from_secs(SESSION_TTL_SECS);
        self.sessions
            .retain(|_, stored| stored.last_accessed.elapsed() < ttl);
    }
}

fn materialize_history(node: &Arc<SessionNode>) -> Vec<Value> {
    let mut chain = Vec::new();
    let mut current = Some(node.clone());
    while let Some(node) = current {
        chain.push(node.clone());
        current = node.parent.clone();
    }

    let capacity = chain
        .iter()
        .map(|node| node.input_delta.len() + node.response_output.len())
        .sum();
    let mut history = Vec::with_capacity(capacity);
    for node in chain.into_iter().rev() {
        history.extend(node.input_delta.iter().cloned());
        history.extend(node.response_output.iter().cloned());
    }
    history
}

static STORE: OnceLock<Mutex<HttpSessionStore>> = OnceLock::new();

fn store() -> &'static Mutex<HttpSessionStore> {
    STORE.get_or_init(|| Mutex::new(HttpSessionStore::new()))
}

/// 根据 previous_response_id 查找历史会话
pub async fn get_session(previous_response_id: &str) -> Option<HttpSessionEntry> {
    store()
        .lock()
        .await
        .get(previous_response_id)
        .map(|(entry, _)| entry)
}

pub async fn get_session_with_parent(
    previous_response_id: &str,
) -> Option<(HttpSessionEntry, SessionParent)> {
    store().lock().await.get(previous_response_id)
}

/// 保存新的会话状态（以 response_id 为 key）
pub async fn save_session(response_id: String, entry: HttpSessionEntry) {
    store().lock().await.insert(response_id, entry);
}

/// 保存 Responses 本轮增量；父节点的 Arc 强引用保证分支共享祖先。
pub async fn save_session_delta(
    response_id: String,
    parent: Option<SessionParent>,
    input_delta: Vec<Value>,
    response_output: Vec<Value>,
    instructions: String,
    model: String,
    routing_session_id: String,
) {
    save_session_delta_with_identity(
        response_id,
        parent,
        input_delta,
        response_output,
        instructions,
        model,
        routing_session_id,
        None,
    )
    .await;
}

pub async fn save_session_delta_with_identity(
    response_id: String,
    parent: Option<SessionParent>,
    input_delta: Vec<Value>,
    response_output: Vec<Value>,
    instructions: String,
    model: String,
    routing_session_id: String,
    scope_identity: Option<String>,
) {
    store().lock().await.insert_delta_with_identity(
        response_id,
        parent,
        input_delta,
        response_output,
        instructions,
        model,
        Some(routing_session_id),
        scope_identity,
    );
}

pub struct PreparedSessionInput {
    pub merged: Vec<Value>,
    pub delta: Vec<Value>,
    pub reset_parent: bool,
}

// Keep only a bounded identity for media omitted from server-side history. Hash the
// complete part (including MIME, detail, URL, etc.) independently of JSON key order.
pub(crate) fn media_placeholder(part: &serde_json::Map<String, Value>) -> Option<Value> {
    use sha2::{Digest, Sha256};
    let label = match part.get("type")?.as_str()? {
        "input_image" | "image_url" => "[historical image omitted]",
        "input_audio" | "audio" | "audio_url" => "[historical audio omitted]",
        _ => return None,
    };
    fn hash_map(map: &serde_json::Map<String, Value>, h: &mut Sha256) {
        h.update(b"{");
        let mut keys: Vec<_> = map.keys().collect();
        keys.sort_unstable();
        for key in keys {
            serde_json::to_writer(&mut *h, key).expect("hash writer");
            hash_value(&map[key], h);
        }
        h.update(b"}");
    }
    fn hash_value(v: &Value, h: &mut Sha256) {
        match v {
            Value::Object(m) => hash_map(m, h),
            Value::Array(a) => {
                h.update(b"[");
                for v in a {
                    hash_value(v, h);
                    h.update(b",");
                }
                h.update(b"]");
            }
            _ => serde_json::to_writer(h, v).expect("hash writer"),
        }
    }
    let mut h = Sha256::new();
    hash_map(part, &mut h);
    Some(
        serde_json::json!({"type":"input_text","text":label,"_am_media_digest":format!("{:x}",h.finalize())}),
    )
}

// A fresh transcript may contain the original media instead of our bounded
// placeholder. Never infer equality from the generic omission text.
fn same_replay_content(old: &Value, new: &Value) -> bool {
    if old == new {
        return true;
    }
    match (old, new) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_replay_content(x, y))
        }
        (Value::Object(a), Value::Object(b)) if a.contains_key("_am_media_digest") => {
            media_placeholder(b).as_ref() == Some(old)
        }
        _ => false,
    }
}

/// 合并请求历史，并在客户端回放完整历史时仅提取新增项。
pub fn prepare_session_input(
    history: Vec<Value>,
    new_input: Vec<Value>,
    tool_call_cache: &HashMap<String, Value>,
) -> PreparedSessionInput {
    prepare_session_input_with_storage(history, new_input, tool_call_cache, true)
}

/// Restore the complete model input while optionally retaining a delta for storage.
pub fn prepare_session_input_with_storage(
    mut history: Vec<Value>,
    new_input: Vec<Value>,
    tool_call_cache: &HashMap<String, Value>,
    retain_delta: bool,
) -> PreparedSessionInput {
    let reset_parent = new_input.iter().any(|item| {
        matches!(
            item.get("type").and_then(Value::as_str),
            Some("compaction") | Some("compaction_summary")
        )
    });
    // A previous_response_id supplies a parent, so inputs are deltas unless we
    // can identify a replay of the parent's prefix or an overlapping suffix.
    // The number of new messages alone says nothing about their provenance.
    fn item_id(item: &Value) -> Option<&str> {
        item.get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    }
    let same_item = |old: &Value, new: &Value| {
        if old == new {
            return true;
        }
        if let (Some(old_id), Some(new_id)) = (item_id(old), item_id(new)) {
            return old_id == new_id;
        }
        // Only messages have role/content semantics. Tool outputs with different
        // call_ids and arguments must not match just because neither has content.
        let message = |v: &Value| {
            v.get("role").is_some()
                && v.get("type").is_none_or(|ty| ty == "message")
                && v.get("content").is_some()
        };
        message(old)
            && message(new)
            && old["role"] == new["role"]
            && same_replay_content(&old["content"], &new["content"])
    };
    let mut replayed = 0;
    if !reset_parent && !history.is_empty() {
        // A real full replay is common and must not grow the chain exponentially.
        if new_input.len() >= history.len()
            && (history.len() > 1
                || item_id(&history[0]).is_some_and(|id| item_id(&new_input[0]) == Some(id)))
            && history.iter().zip(&new_input).all(|(a, b)| same_item(a, b))
        {
            replayed = history.len();
        } else if let Some(first_id) = new_input.first().and_then(item_id) {
            // Partial overlap is safe only when the incoming prefix starts at a
            // known item ID. Repeated unlabelled text is a valid new turn. IDs in
            // stored history are unique; if a malformed parent repeats this ID,
            // treat the boundary as ambiguous rather than dropping input.
            let mut matches = history
                .iter()
                .enumerate()
                .filter(|(_, item)| item_id(item) == Some(first_id));
            if let Some((start, _)) = matches.next() {
                let suffix = &history[start..];
                if matches.next().is_none()
                    && suffix.len() <= new_input.len()
                    && suffix.iter().zip(&new_input).all(|(a, b)| same_item(a, b))
                {
                    replayed = suffix.len();
                }
            }
        }
    }
    let mut incoming = new_input.into_iter();
    let start = history.len() - replayed;
    for (old, mut new) in history[start..].iter_mut().zip(incoming.by_ref()) {
        // Only upgrade verified, equivalent content; ID-based deduplication alone
        // does not authorize replacing an existing message with edited content.
        if old.get("content").is_some()
            && old["content"] != new["content"]
            && same_replay_content(&old["content"], &new["content"])
        {
            old["content"] = new["content"].take();
        }
    }
    let new_items: Vec<Value> = incoming.collect();
    if !retain_delta {
        let merged = merge_history_with_new_input(
            if reset_parent { Vec::new() } else { history },
            &[],
            new_items,
            tool_call_cache,
        );
        return PreparedSessionInput {
            merged,
            delta: Vec::new(),
            reset_parent,
        };
    }
    if !reset_parent && new_items.is_empty() {
        return PreparedSessionInput {
            merged: history,
            delta: Vec::new(),
            reset_parent: false,
        };
    }
    let parent_len = history.len();
    let merged = merge_history_with_new_input(
        if reset_parent {
            Vec::new()
        } else {
            history.clone()
        },
        &[],
        new_items,
        tool_call_cache,
    );
    // Tool repair/id deduplication may also replace an earlier item. In that
    // case persist a new root rather than reconstructing a different history.
    let reset_parent = reset_parent || !merged.starts_with(&history);
    let delta = if reset_parent {
        merged.clone()
    } else {
        merged[parent_len..].to_vec()
    };

    PreparedSessionInput {
        merged,
        delta,
        reset_parent,
    }
}

/// 把上一轮的 response output items 转成 input items 追加到历史中
/// 同时把新的 user input items 追加进去
/// 返回合并后的 input items
pub fn merge_history_with_new_input(
    mut history: Vec<Value>,
    response_output: &[Value],
    new_input: Vec<Value>,
    tool_call_cache: &HashMap<String, Value>,
) -> Vec<Value> {
    // 检测新输入中是否包含 compaction / compaction_summary，如果包含，说明客户端正在发送压缩后的全新完整历史
    let has_compaction = new_input.iter().any(|item| {
        let t = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        t == "compaction" || t == "compaction_summary"
    });

    if has_compaction {
        tracing::info!(
            "[Session] Compaction detected in new input. Overwriting stale history (new items: {})",
            new_input.len()
        );
        // 过滤掉 compaction 本身
        let mut filtered = Vec::new();
        for item in new_input {
            let t = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if t == "compaction" || t == "compaction_summary" {
                continue;
            }
            filtered.push(item);
        }
        repair_tool_calls(&mut filtered, tool_call_cache);
        return dedupe_input_items(filtered);
    }

    // 追加上一轮 response output（assistant消息、工具调用等）
    for item in response_output {
        history.push(item.clone());
    }

    // 追加新的 input items
    for item in new_input {
        let t = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if t == "compaction" || t == "compaction_summary" {
            continue;
        }
        history.push(item);
    }

    // 修复工具调用（确保function_call_output前有对应的function_call）
    repair_tool_calls(&mut history, tool_call_cache);

    // 去重
    dedupe_input_items(history)
}

fn repair_tool_calls(items: &mut Vec<Value>, tool_call_cache: &HashMap<String, Value>) {
    let mut call_present = std::collections::HashSet::new();
    for item in items.iter() {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "function_call" || item_type == "custom_tool_call" {
            if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                call_present.insert(call_id.to_string());
            }
        }
    }

    let mut new_items = Vec::new();
    let mut inserted = std::collections::HashSet::new();
    for item in items.drain(..) {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "function_call_output" || item_type == "custom_tool_call_output" {
            if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                if !call_id.is_empty()
                    && !call_present.contains(call_id)
                    && !inserted.contains(call_id)
                {
                    if let Some(cached_call) = tool_call_cache
                        .get(call_id)
                        .cloned()
                        .or_else(|| get_cached_tool_call(call_id))
                    {
                        new_items.push(cached_call.clone());
                        inserted.insert(call_id.to_string());
                    }
                }
            }
        }
        new_items.push(item);
    }
    *items = new_items;
}

fn dedupe_input_items(items: Vec<Value>) -> Vec<Value> {
    use std::collections::{HashMap, HashSet};
    let mut referenced_call_ids = HashSet::new();
    for item in &items {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "function_call_output" || item_type == "custom_tool_call_output" {
            if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                if !call_id.is_empty() {
                    referenced_call_ids.insert(call_id.to_string());
                }
            }
        }
    }

    let mut keep_map: HashMap<String, usize> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if item_id.is_empty() {
            continue;
        }
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let is_referenced = !call_id.is_empty() && referenced_call_ids.contains(call_id);
        if let Some(&existing_idx) = keep_map.get(item_id) {
            let existing_call_id = items[existing_idx]
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let existing_referenced =
                !existing_call_id.is_empty() && referenced_call_ids.contains(existing_call_id);
            if is_referenced || !existing_referenced {
                keep_map.insert(item_id.to_string(), idx);
            }
        } else {
            keep_map.insert(item_id.to_string(), idx);
        }
    }

    let mut keep_indices = std::collections::HashSet::new();
    for (_, idx) in keep_map {
        keep_indices.insert(idx);
    }

    let mut filtered = Vec::new();
    for (idx, item) in items.into_iter().enumerate() {
        let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if !item_id.is_empty() && !keep_indices.contains(&idx) {
            continue;
        }
        filtered.push(item);
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(text: &str) -> HttpSessionEntry {
        HttpSessionEntry {
            input_items: vec![json!({
                "id": format!("msg-{text}"),
                "type": "message",
                "role": "user",
                "content": text
            })],
            instructions: "be concise".to_string(),
            model: "gemini-3.7-flash-high".to_string(),
            last_accessed: Instant::now(),
        }
    }

    #[test]
    fn bounded_media_matches_and_restores_only_the_verified_full_replay() {
        let plain = |text: &str| json!({"type":"input_text","text":text});
        let image =
            json!({"type":"input_image","image_url":"data:image/png;base64,AQ==","detail":"high"});
        let old = json!({"role":"user","content":[plain("look"),media_placeholder(image.as_object().unwrap()).unwrap()]});
        let answer = json!({"role":"assistant","id":"a","content":"answer"});
        let next = json!({"role":"user","content":"again"});
        for keep in [true, false] {
            let original = json!({"role":"user","content":[plain("look"),{"detail":"high","image_url":image["image_url"],"type":"input_image"}]});
            let parent = vec![old.clone(), answer.clone()];
            let restored = prepare_session_input_with_storage(
                parent.clone(),
                vec![original.clone(), answer.clone(), next.clone()],
                &Default::default(),
                keep,
            );
            assert_eq!(
                restored.merged,
                vec![original.clone(), answer.clone(), next.clone()]
            );
            assert!(!restored.reset_parent);
            assert_eq!(
                restored.delta,
                if keep { vec![next.clone()] } else { vec![] }
            );
            let mut changed = original;
            changed["content"][1]["image_url"] = json!("data:image/png;base64,Ag==");
            let other = prepare_session_input_with_storage(
                parent,
                vec![changed.clone(), answer.clone(), next.clone()],
                &Default::default(),
                keep,
            );
            assert!(other.merged.contains(&old));
            assert!(other.merged.contains(&changed));
        }
    }

    #[test]
    fn responses_store_false_restores_full_input_without_retaining_delta() {
        let history = vec![
            json!({"id": "old", "role": "user", "content": "x".repeat(32768)}),
            json!({"id": "answer", "role": "assistant", "content": "prior answer"}),
        ];
        let next = json!({"id": "new", "role": "user", "content": "y".repeat(32768)});
        let mut full_input = history.clone();
        full_input.push(next.clone());
        for replay in [vec![next], full_input.clone()] {
            let prepared =
                prepare_session_input_with_storage(history.clone(), replay, &HashMap::new(), false);
            assert!(prepared.delta.is_empty());
            assert_eq!(prepared.merged, full_input);
        }
        let prepared =
            prepare_session_input_with_storage(Vec::new(), history.clone(), &HashMap::new(), false);
        assert!(prepared.delta.is_empty());
        assert_eq!(prepared.merged, history);
    }

    #[test]
    fn responses_store_false_full_tool_replay_needs_no_global_cache() {
        let call_id = format!("uncached-{}", uuid::Uuid::new_v4());
        let input = vec![
            json!({"id":"call-item", "type":"function_call", "call_id":call_id, "name":"shell_command", "arguments":"{\"command\":\"pwd\"}"}),
            json!({"type":"function_call_output", "call_id":call_id, "output":"/synthetic/workspace"}),
            json!({"role":"user", "content":"continue"}),
        ];
        assert!(get_cached_tool_call(&call_id).is_none());
        let prepared =
            prepare_session_input_with_storage(Vec::new(), input.clone(), &HashMap::new(), false);
        assert!(prepared.delta.is_empty());
        assert_eq!(prepared.merged, input);
    }

    #[test]
    fn session_chain_stores_delta_and_materializes_history() {
        let mut store = HttpSessionStore::new();
        store.insert("resp-1".to_string(), entry("first"));
        let (root, parent) = store.get("resp-1").expect("root");
        let mut replay = root.input_items.clone();
        replay.push(json!({"id": "msg-second", "content": "second"}));
        let prepared = prepare_session_input(root.input_items, replay, &HashMap::new());
        assert_eq!(prepared.delta.len(), 1);
        assert_eq!(prepared.merged.len(), 2);
        store.insert_delta(
            "resp-2".to_string(),
            Some(parent),
            prepared.delta,
            vec![json!({"id": "out-second", "content": "answer"})],
            "be concise".to_string(),
            "gemini-3.7-flash-high".to_string(),
            Some("routing-root".to_string()),
        );

        let (previous, _) = store.get("resp-2").expect("child");
        assert_eq!(previous.input_items[0]["content"], "first");
        assert_eq!(previous.input_items.len(), 3);
        assert_eq!(store.sessions["resp-2"].node.input_delta.len(), 1);
        assert_eq!(store.sessions["resp-2"].node.response_output.len(), 1);
    }

    #[test]
    fn old_response_id_branches_share_parent() {
        let mut store = HttpSessionStore::new();
        store.insert("resp-root".to_string(), entry("root"));
        let (_, parent_a) = store.get("resp-root").expect("parent a");
        let (_, parent_b) = store.get("resp-root").expect("parent b");
        assert!(Arc::ptr_eq(&parent_a.0, &parent_b.0));

        store.insert_delta(
            "resp-a".to_string(),
            Some(parent_a),
            vec![json!({"content": "branch a"})],
            Vec::new(),
            String::new(),
            String::new(),
            Some("routing-root".to_string()),
        );
        store.insert_delta(
            "resp-b".to_string(),
            Some(parent_b),
            vec![json!({"content": "branch b"})],
            Vec::new(),
            String::new(),
            String::new(),
            Some("routing-root".to_string()),
        );

        let parent_a = store.sessions["resp-a"].node.parent.as_ref().unwrap();
        let parent_b = store.sessions["resp-b"].node.parent.as_ref().unwrap();
        assert!(Arc::ptr_eq(parent_a, parent_b));
        assert_eq!(
            store.sessions["resp-a"].node.routing_session_id,
            "routing-root"
        );
        assert_eq!(
            store.sessions["resp-b"].node.routing_session_id,
            "routing-root"
        );
    }

    #[test]
    fn prepare_session_input_prevents_duplication_on_full_history_replay_without_ids() {
        // Test case when client resends full history without IDs:
        // history has 2 messages, new_input has 3 messages (the same 2 + 1 new), but no "id" field.
        let history = vec![
            json!({"role": "user", "type": "message", "content": "hello"}),
            json!({"role": "assistant", "type": "message", "content": "hi there"}),
        ];
        let new_input = vec![
            json!({"role": "user", "type": "message", "content": "hello"}),
            json!({"role": "assistant", "type": "message", "content": "hi there"}),
            json!({"role": "user", "type": "message", "content": "next question"}),
        ];

        let prepared = prepare_session_input(history, new_input, &HashMap::new());
        // Delta should be the 1 new message, not all 3 messages!
        assert_eq!(prepared.delta.len(), 1);
        assert_eq!(prepared.delta[0]["content"], "next question");
        // Merged history should be 3 messages, NOT 5 (2 + 3)!
        assert_eq!(prepared.merged.len(), 3);
    }

    #[test]
    fn prepare_session_input_preserves_unmatched_delta_even_when_long_as_parent() {
        let history = vec![
            json!({"role": "user", "type": "message", "content": "msg 1"}),
            json!({"role": "assistant", "type": "message", "content": "msg 2"}),
        ];
        // Client sends 2 different messages (length >= history)
        let new_input = vec![
            json!({"role": "user", "type": "message", "content": "different 1"}),
            json!({"role": "assistant", "type": "message", "content": "different 2"}),
        ];

        let prepared = prepare_session_input(history.clone(), new_input.clone(), &HashMap::new());
        assert_eq!(prepared.delta, new_input);
        assert_eq!(prepared.merged, [history, new_input].concat());
        assert!(!prepared.reset_parent);
    }
}

#[cfg(test)]
mod continuation_delta_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn model_input_and_restored_child_agree_for_deltas_replays_and_compaction() {
        let root = vec![
            json!({"role":"user","type":"message","content":"Task","id":"u"}),
            json!({"role":"assistant","type":"message","content":"Answer","id":"a"}),
        ];
        let extra = vec![
            json!({"role":"developer","content":"New instruction"}),
            json!({"role":"user","content":"Continue"}),
        ];
        let expected = [root.clone(), extra.clone()].concat();
        let mut semantic = root.clone();
        for item in &mut semantic {
            item.as_object_mut().unwrap().remove("id");
        }
        semantic.extend(extra.clone());
        let reset = vec![json!({"type":"compaction_summary"}), extra[1].clone()];
        for (label, new, expected, detach) in [
            ("equal length delta", extra.clone(), expected.clone(), false),
            (
                "longer delta",
                [
                    extra.clone(),
                    vec![json!({"role":"user","content":"Extra"})],
                ]
                .concat(),
                [
                    expected.clone(),
                    vec![json!({"role":"user","content":"Extra"})],
                ]
                .concat(),
                false,
            ),
            ("exact replay", expected.clone(), expected.clone(), false),
            (
                "same content without IDs",
                semantic,
                expected.clone(),
                false,
            ),
            (
                "overlapping tail",
                [vec![root[1].clone()], extra.clone()].concat(),
                expected,
                false,
            ),
            ("explicit compaction", reset, vec![extra[1].clone()], true),
        ] {
            for retain in [false, true] {
                let mut store = HttpSessionStore::new();
                store.insert_delta(
                    "parent".into(),
                    None,
                    root.clone(),
                    vec![],
                    String::new(),
                    "m".into(),
                    None,
                );
                let (entry, parent) = store.get("parent").unwrap();
                let prep = prepare_session_input_with_storage(
                    entry.input_items,
                    new.clone(),
                    &Default::default(),
                    retain,
                );
                assert_eq!(prep.merged, expected, "{label}");
                assert_eq!(prep.reset_parent, detach, "{label}");
                if retain {
                    store.insert_delta(
                        "child".into(),
                        (!prep.reset_parent).then_some(parent),
                        prep.delta,
                        vec![],
                        String::new(),
                        "m".into(),
                        None,
                    );
                    assert_eq!(
                        store.get("child").unwrap().0.input_items,
                        expected,
                        "{label}"
                    );
                } else {
                    assert!(prep.delta.is_empty())
                }
            }
        }
    }
    #[test]
    fn repeated_messages_only_overlap_at_a_known_boundary() {
        let a = json!({"role":"user","content":"A"});
        let b = json!({"role":"assistant","content":"B"});
        let c = json!({"role":"user","content":"C"});
        let with_id = |mut item: Value, id: &str| {
            item["id"] = json!(id);
            item
        };
        let b1 = with_id(b.clone(), "first-b");
        let b2 = with_id(b.clone(), "second-b");
        for (parent, new, delta) in [
            (
                vec![a.clone(), b1.clone()],
                vec![b1.clone(), b2.clone(), c.clone()],
                vec![b2.clone(), c.clone()],
            ),
            (
                vec![a.clone(), b1.clone()],
                vec![b2.clone(), c.clone()],
                vec![b2.clone(), c.clone()],
            ),
            (vec![b1.clone()], vec![b2.clone()], vec![b2]),
            (vec![b.clone()], vec![b.clone()], vec![b.clone()]),
            (
                vec![a, b.clone()],
                vec![b.clone(), b.clone(), c.clone()],
                vec![b.clone(), b, c],
            ),
        ] {
            for retain in [true, false] {
                let prepared = prepare_session_input_with_storage(
                    parent.clone(),
                    new.clone(),
                    &HashMap::new(),
                    retain,
                );
                assert_eq!(
                    prepared.merged,
                    [parent.as_slice(), delta.as_slice()].concat()
                );
                if retain {
                    assert_eq!(prepared.delta, delta);
                } else {
                    assert!(prepared.delta.is_empty());
                }
            }
        }
    }
    #[test]
    fn distinct_tool_items_are_not_semantic_replays() {
        let first = json!({"type":"function_call_output","call_id":"one","output":"old value"});
        let next = json!({"type":"function_call_output","call_id":"two","output":"new value"});
        let prep =
            prepare_session_input(vec![first.clone()], vec![next.clone()], &Default::default());
        assert_eq!(prep.merged, vec![first, next.clone()]);
        assert_eq!(prep.delta, vec![next]);
    }
}
