use super::convert_codex_to_openai_request;
use crate::proxy::mappers::{claude, openai};
use serde_json::{json, Value};

fn texts(value: &Value) -> Vec<&str> {
    let mut result = Vec::new();
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "text" {
                    if let Some(text) = value.as_str() {
                        result.push(text);
                    }
                }
                result.extend(texts(value));
            }
        }
        Value::Array(values) => {
            for value in values {
                result.extend(texts(value));
            }
        }
        _ => {}
    }
    result
}

fn chat(body: Value, responses: bool) -> Value {
    let request: openai::models::OpenAIRequest = serde_json::from_value(body).unwrap();
    openai::request::transform_openai_request_with_session(
        &request,
        "r30-project",
        "gemini-2.5-flash",
        None,
        "r30-readable-history",
        None,
        responses,
        false,
    )
    .unwrap()
    .0
}

fn anthropic(body: Value) -> Value {
    let request: claude::models::ClaudeRequest = serde_json::from_value(body).unwrap();
    claude::transform_claude_request_in(
        &request,
        "r30-project",
        false,
        None,
        "r30-readable-history",
        None,
    )
    .unwrap()
}

#[test]
fn r30_user_text_after_tool_result_preserves_whitespace_changes_and_repeat_requests() {
    let first = "Write \"A B\" to the file.";
    for next in [
        "Write \"AB\" to the file.",
        "Write \"A\tB\" to the file.",
        first,
    ] {
        let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
            {"role":"user","content":[{"type":"text","text":first}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_write","name":"write_file","input":{"value":"A B"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_write","content":"WROTE_FIRST_STRING"},{"type":"text","text":next}]}
        ]});
        let mapped = anthropic(body);
        let text = texts(&mapped);
        assert_eq!(
            text.iter().filter(|text| **text == first).count(),
            if next == first { 2 } else { 1 }
        );
        assert_eq!(text.last().copied(), Some(next));
        assert!(mapped.to_string().contains("WROTE_FIRST_STRING"));
    }
}

#[test]
fn r30_responses_markdown_thinking_heading_remains_visible_history() {
    for phase in ["commentary", "final_answer"] {
        let body = json!({"model":"gemini-2.5-flash","input":[
            {"role":"user","content":"Give the book title and code."},
            {"type":"message","id":"msg_book_answer","role":"assistant","status":"completed","phase":phase,
                "content":[{"type":"output_text","text":"**Thinking** is the title; the code is BOOK_42.","annotations":[]}]},
            {"role":"user","content":"Repeat the catalog code."}
        ]});
        let mapped = chat(convert_codex_to_openai_request(body.clone()), true);
        assert!(texts(&mapped).iter().any(|text| text.contains("BOOK_42")));
        let mut private_display = body;
        private_display["input"][1]["id"] = json!("msg_thought_local_42");
        let mapped = chat(convert_codex_to_openai_request(private_display), true);
        assert!(!texts(&mapped).iter().any(|text| text.contains("BOOK_42")));
    }
}

#[test]
fn r30_chat_refusal_blocks_preserve_order_and_match_top_level_refusal() {
    let body = json!({"model":"gemini-2.5-flash","messages":[
        {"role":"user","content":"Original request"},
        {"role":"assistant","content":[
            {"type":"text","text":"BEFORE_REFUSAL"},
            {"type":"refusal","refusal":"REFUSAL_BODY_42: I can offer a safe alternative."},
            {"type":"text","text":"AFTER_REFUSAL"}
        ]},
        {"role":"user","content":"Explain the alternative."}
    ]});
    let mapped = chat(body.clone(), false);
    let text = texts(&mapped);
    let positions: Vec<_> = [
        "BEFORE_REFUSAL",
        "REFUSAL_BODY_42: I can offer a safe alternative.",
        "AFTER_REFUSAL",
    ]
    .iter()
    .map(|needle| text.iter().position(|text| text == needle).unwrap())
    .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    let mut field = body;
    field["messages"][1] = json!({"role":"assistant","content":null,"refusal":"REFUSAL_BODY_42: I can offer a safe alternative."});
    assert!(
        texts(&chat(field, false)).contains(&"REFUSAL_BODY_42: I can offer a safe alternative.")
    );
}

#[test]
fn r30_tool_search_history_preserves_names_and_errors_without_local_tool_execution() {
    for content in [
        json!({"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"weather_lookup"},{"type":"tool_reference","tool_name":"calendar_read"}]}),
        json!({"type":"tool_search_tool_search_result","tool_references":[]}),
        json!({"type":"tool_search_tool_result_error","error_code":"unavailable","error_message":"SEARCH_ERROR_42"}),
    ] {
        let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
            {"role":"user","content":"Find the tools"},
            {"role":"assistant","content":[
                {"type":"text","text":"BEFORE_SEARCH"},
                {"type":"server_tool_use","id":"srvtoolu_search","name":"tool_search_tool_regex","input":{"pattern":"weather"}},
                {"type":"tool_search_tool_result","tool_use_id":"srvtoolu_search","content":content},
                {"type":"text","text":"AFTER_SEARCH"}
            ]},
            {"role":"user","content":"Continue using the available context."}
        ]});
        let mapped = anthropic(body);
        let text = texts(&mapped);
        let start = text
            .iter()
            .position(|text| *text == "BEFORE_SEARCH")
            .unwrap();
        let result = text
            .iter()
            .position(|text| text.contains("srvtoolu_search"))
            .unwrap();
        let end = text
            .iter()
            .position(|text| *text == "AFTER_SEARCH")
            .unwrap();
        assert!(start < result && result < end);
        for reference in content["tool_references"].as_array().into_iter().flatten() {
            assert!(text
                .iter()
                .any(|text| text.contains(reference["tool_name"].as_str().unwrap())));
        }
        if let Some(message) = content["error_message"].as_str() {
            assert!(text.contains(&message));
            assert!(text.iter().any(|text| text.contains("unavailable")));
        }
        assert!(!mapped.to_string().contains("functionCall"));
        assert!(!mapped.to_string().contains("functionResponse"));
    }
}

#[test]
fn r30_known_history_blocks_do_not_admit_malformed_envelopes_or_tool_arguments() {
    for block in [
        json!({"type":"refusal"}),
        json!({"type":"refusal","refusal":42}),
        json!({"type":"unknown_text_kind","text":"not a known content block"}),
    ] {
        assert!(serde_json::from_value::<openai::models::OpenAIRequest>(
            json!({"model":"g","messages":[{"role":"assistant","content":[block]}]})
        )
        .is_err());
    }
    for content in [
        json!({"type":"tool_search_tool_search_result"}),
        json!({"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference"}]}),
        json!({"type":"tool_search_tool_search_result","tool_references":[{"type":"text","text":"unexpected"}]}),
        json!({"type":"tool_search_tool_result_error"}),
        json!({"type":"tool_search_tool_result_error","error_code":42}),
        json!({"type":"tool_use","id":"local","name":"bad"}),
    ] {
        assert!(serde_json::from_value::<claude::models::ClaudeRequest>(json!({"model":"g","messages":[{"role":"assistant","content":[{"type":"tool_search_tool_result","tool_use_id":"srvtoolu","content":content}]}]})).is_err());
    }
    assert!(serde_json::from_value::<claude::models::ClaudeRequest>(json!({"model":"g","messages":[{"role":"assistant","content":[{"type":"tool_search_tool_result","content":{"type":"tool_search_tool_search_result","tool_references":[]}}]}]})).is_err());
    assert!(serde_json::from_value::<openai::models::OpenAIRequest>(json!({"model":"g","messages":[{"role":"assistant","tool_calls":[{"id":"call_bad","type":"custom","custom":{"name":"script"}}]}]})).is_err());
    assert!(serde_json::from_value::<openai::models::OpenAIRequest>(json!({"model":"g","messages":[{"role":"assistant","tool_calls":[{"id":"call_bad","type":"function","function":{"name":"write_file"}}]}]})).is_err());
    assert!(serde_json::from_value::<claude::models::ClaudeRequest>(json!({"model":"g","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_bad","input":{}}]}]})).is_err());
}
