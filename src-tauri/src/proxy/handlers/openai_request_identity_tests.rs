use super::{convert_codex_to_openai_request, validate_responses_continuation};
use crate::proxy::mappers::{claude, gemini, openai};
use serde_json::{json, Value};

fn chat(body: Value, responses: bool) -> Value {
    let request: openai::models::OpenAIRequest = serde_json::from_value(body).unwrap();
    openai::request::transform_openai_request_with_session(
        &request,
        "r28-project",
        "gemini-2.5-flash",
        None,
        "r28-request-identity",
        None,
        responses,
        false,
    )
    .unwrap()
    .0
}

fn values_at<'a>(value: &'a Value, key: &str) -> Vec<&'a Value> {
    let mut values = Vec::new();
    match value {
        Value::Object(object) => {
            for (name, value) in object {
                if name == key {
                    values.push(value);
                }
                values.extend(values_at(value, key));
            }
        }
        Value::Array(array) => {
            for value in array {
                values.extend(values_at(value, key));
            }
        }
        _ => {}
    }
    values
}

#[test]
fn r28_chat_nested_custom_keeps_declaration_selector_history_and_raw_input() {
    let body = json!({"model":"gemini-2.5-flash","tools":[{"type":"custom","custom":{
        "name":"run_script","description":"Run script", "format":{"type":"grammar","grammar":{"syntax":"lark","definition":"start: WORD"}}
    }}],"tool_choice":{"type":"custom","custom":{"name":"run_script"}},"messages":[
        {"role":"user","content":"Run the script"},
        {"role":"assistant","content":null,"tool_calls":[{"id":"call_script","type":"custom","custom":{"name":"run_script","input":"print('CUSTOM_INPUT_42')\n"}}]},
        {"role":"tool","tool_call_id":"call_script","content":"CUSTOM_RESULT_42"}
    ]});
    let kinds = openai::request::ResponsesToolKinds::from_tools(
        body["tools"].as_array().map(Vec::as_slice),
    );
    let mapped = chat(body.clone(), false);
    let declarations = values_at(&mapped, "functionDeclarations");
    assert_eq!(declarations[0][0]["name"], "run_script");
    assert_eq!(
        declarations[0][0]["parameters"]["required"],
        json!(["input"])
    );
    assert!(declarations[0][0]["description"]
        .as_str()
        .unwrap()
        .contains("start: WORD"));
    assert_eq!(
        values_at(&mapped, "allowedFunctionNames"),
        vec![&json!(["run_script"])]
    );
    let calls = values_at(&mapped, "functionCall");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["args"],
        json!({"input":"print('CUSTOM_INPUT_42')\n"})
    );
    let results = values_at(&mapped, "functionResponse");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["name"], "run_script");
    assert_eq!(results[0]["id"], calls[0]["id"]);
    assert!(results[0].to_string().contains("CUSTOM_RESULT_42"));
    assert_eq!(
        kinds.wire_identity("run_script"),
        Some(("run_script".into(), None, true))
    );
    for invalid in [
        json!({"name":"run_script"}),
        json!({"name":"run_script","input":{}}),
    ] {
        let mut invalid_body = body.clone();
        invalid_body["messages"][1]["tool_calls"][0]["custom"] = invalid;
        assert!(serde_json::from_value::<openai::models::OpenAIRequest>(invalid_body).is_err());
    }
}

fn colliding_tools() -> Value {
    json!([
        {"type":"namespace","name":"a","description":"First","tools":[{"type":"function","name":"b__c","description":"FIRST_TOOL","parameters":{"type":"object"}}]},
        {"type":"namespace","name":"a__b","description":"Second","tools":[{"type":"custom","name":"c","description":"SECOND_TOOL"}]}
    ])
}

#[test]
fn r28_namespace_collisions_preserve_selector_history_and_return_identity() {
    let tools = colliding_tools();
    let kinds =
        openai::request::ResponsesToolKinds::from_tools(tools.as_array().map(Vec::as_slice));
    let first = kinds.backend_name("b__c", Some("a"));
    let second = kinds.backend_name("c", Some("a__b"));
    assert_ne!(first, second);
    for (namespace, name, kind, backend) in [
        ("a", "b__c", "function", &first),
        ("a__b", "c", "custom", &second),
    ] {
        let body = json!({"model":"gemini-2.5-flash","tools":tools,
            "tool_choice":{"type":kind,"name":name,"namespace":namespace},"input":[
            {"role":"user","content":"Use both tools"},
            {"type":"function_call","call_id":"call_first","name":"b__c","namespace":"a","arguments":"{}"},
            {"type":"custom_tool_call","call_id":"call_second","name":"c","namespace":"a__b","input":"RAW_SECOND_INPUT"},
            {"type":"function_call_output","call_id":"call_first","output":"FIRST_RESULT"},
            {"type":"custom_tool_call_output","call_id":"call_second","output":"SECOND_RESULT"}
        ]});
        assert!(validate_responses_continuation(&body).is_ok());
        let mapped = chat(convert_codex_to_openai_request(body), true);
        assert_eq!(
            values_at(&mapped, "allowedFunctionNames"),
            vec![&json!([backend])]
        );
        let declarations: Vec<_> = values_at(&mapped, "functionDeclarations")
            .into_iter()
            .filter_map(Value::as_array)
            .flatten()
            .collect();
        assert_eq!(declarations.len(), 2);
        assert_eq!(declarations[0]["name"], first);
        assert_eq!(declarations[1]["name"], second);
        let calls = values_at(&mapped, "functionCall");
        let results = values_at(&mapped, "functionResponse");
        assert_eq!(calls.len(), 2);
        assert_eq!(results.len(), 2);
        for expected in [&first, &second] {
            let call = calls.iter().find(|call| call["name"] == *expected).unwrap();
            let result = results
                .iter()
                .find(|result| result["id"] == call["id"])
                .unwrap();
            assert_eq!(result["name"], *expected);
        }
        let args = json!({"input":"RAW_RETURN_INPUT"});
        let returned = openai::streaming::responses_tool_call_item(
            backend,
            Some(&args),
            &args.to_string(),
            "item",
            "call",
            &kinds,
        );
        assert_eq!(returned["name"], name);
        assert_eq!(returned["namespace"], namespace);
        assert_eq!(
            returned["type"],
            if kind == "custom" {
                "custom_tool_call"
            } else {
                "function_call"
            }
        );
    }
    let all = chat(
        convert_codex_to_openai_request(
            json!({"model":"gemini-2.5-flash","tools":tools,"input":"Use both tools"}),
        ),
        true,
    );
    let declarations: Vec<_> = values_at(&all, "functionDeclarations")
        .into_iter()
        .filter_map(Value::as_array)
        .flatten()
        .collect();
    assert_eq!(declarations.len(), 2);
    assert_eq!(declarations[0]["name"], first);
    assert_eq!(declarations[1]["name"], second);
    assert_eq!(
        kinds.qualified_names_in_namespace("a"),
        vec![first.as_str()]
    );
}

#[test]
fn r28_namespace_aliases_are_order_independent_bounded_and_do_not_steal_literal_names() {
    let mut tools = colliding_tools();
    let original =
        openai::request::ResponsesToolKinds::from_tools(tools.as_array().map(Vec::as_slice));
    let first = original.backend_name("b__c", Some("a"));
    tools.as_array_mut().unwrap().reverse();
    let reordered =
        openai::request::ResponsesToolKinds::from_tools(tools.as_array().map(Vec::as_slice));
    assert_eq!(reordered.backend_name("b__c", Some("a")), first);
    tools
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"function","name":first,"parameters":{"type":"object"}}));
    tools.as_array_mut().unwrap().push(json!({"type":"namespace","name":"x".repeat(64),"tools":[{"type":"function","name":"y".repeat(64)}]}));
    let extended =
        openai::request::ResponsesToolKinds::from_tools(tools.as_array().map(Vec::as_slice));
    assert_eq!(extended.backend_name(&first, None), first);
    assert_ne!(extended.backend_name("b__c", Some("a")), first);
    assert!(
        extended
            .backend_name(&"y".repeat(64), Some(&"x".repeat(64)))
            .len()
            <= 64
    );
    let control =
        json!([{"type":"namespace","name":"fs","tools":[{"type":"function","name":"lookup"}]}]);
    let control =
        openai::request::ResponsesToolKinds::from_tools(control.as_array().map(Vec::as_slice));
    assert_eq!(control.backend_name("lookup", Some("fs")), "fs__lookup");
}

#[test]
fn r28_claude_search_result_preserves_readable_context_and_rejects_malformed_blocks() {
    let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[{"role":"user","content":[
        {"type":"text","text":"BEFORE_SEARCH"},
        {"type":"search_result","source":"https://example.test/report","title":"Release report","content":[{"type":"text","text":"RETRIEVED_BODY_42"}]},
        {"type":"text","text":"AFTER_SEARCH"}
    ]}]});
    let request: claude::models::ClaudeRequest = serde_json::from_value(body.clone()).unwrap();
    let mapped = claude::transform_claude_request_in(
        &request,
        "r28-project",
        false,
        None,
        "r28-search",
        None,
    )
    .unwrap();
    let readable = values_at(&mapped, "text")
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for text in [
        "Release report",
        "https://example.test/report",
        "RETRIEVED_BODY_42",
    ] {
        assert!(readable.contains(text));
    }
    assert!(readable.find("BEFORE_SEARCH").unwrap() < readable.find("RETRIEVED_BODY_42").unwrap());
    assert!(readable.find("RETRIEVED_BODY_42").unwrap() < readable.find("AFTER_SEARCH").unwrap());
    for invalid in [
        json!({"type":"text","text":42}),
        json!({"type":"tool_use","id":"bad"}),
    ] {
        let mut invalid_body = body.clone();
        invalid_body["messages"][0]["content"][1]["content"] = json!([invalid]);
        assert!(serde_json::from_value::<claude::models::ClaudeRequest>(invalid_body).is_err());
    }
}

#[test]
fn r28_native_gemini_claude_ids_pair_across_split_results_without_explicit_id_collisions() {
    for explicit_first in [false, true] {
        let mut body = json!({"contents":[
            {"role":"user","parts":[{"text":"Read files"}]},
            {"role":"model","parts":[{"functionCall":{"name":"read","args":{"path":"FIRST_PATH"}}},{"functionCall":{"name":"read","args":{"path":"SECOND_PATH"}}}]},
            {"role":"user","parts":[{"functionResponse":{"name":"read","response":{"output":"FIRST_OUTPUT"}}}]},
            {"role":"user","parts":[{"functionResponse":{"name":"read","response":{"output":"SECOND_OUTPUT"}}}]}
        ]});
        if explicit_first {
            body["contents"][1]["parts"][0]["functionCall"]["id"] = json!("call_read_0");
            body["contents"][2]["parts"][0]["functionResponse"]["id"] = json!("call_read_0");
        }
        let mapped = gemini::wrapper::wrap_request(
            &body,
            "r28-project",
            "claude-sonnet-4-5",
            None,
            None,
            None,
        )
        .unwrap();
        let calls = values_at(&mapped, "functionCall");
        let results = values_at(&mapped, "functionResponse");
        assert_eq!(calls.len(), 2);
        assert_eq!(results.len(), 2);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        for (path, output) in [
            ("FIRST_PATH", "FIRST_OUTPUT"),
            ("SECOND_PATH", "SECOND_OUTPUT"),
        ] {
            let call = calls
                .iter()
                .find(|call| call["args"]["path"] == path)
                .unwrap();
            let result = results
                .iter()
                .find(|result| result["response"]["output"] == output)
                .unwrap();
            assert_eq!(call["id"], result["id"]);
        }
        if explicit_first {
            assert_eq!(calls[0]["id"], "call_read_0");
        }
        let control = gemini::wrapper::wrap_request(
            &body,
            "r28-project",
            "gemini-2.5-flash",
            None,
            None,
            None,
        )
        .unwrap();
        assert!(values_at(&control, "functionCall")[1].get("id").is_none());
    }
}

#[test]
fn r28_anonymous_results_respect_explicit_aliases_and_do_not_cross_user_turns() {
    let body = json!({"contents":[
        {"role":"user","parts":[{"text":"Read files"}]},
        {"role":"model","parts":[
            {"functionCall":{"id":"call12","name":"read","args":{"path":"EXPLICIT"}}},
            {"functionCall":{"name":"read","args":{"path":"ANONYMOUS"}}}
        ]},
        {"role":"user","parts":[{"functionResponse":{"name":"read","response":{"output":"ANONYMOUS_OUTPUT"}}}]},
        {"role":"user","parts":[{"functionResponse":{"id":"call_12","name":"read","response":{"output":"EXPLICIT_OUTPUT"}}}]},
        {"role":"model","parts":[{"functionCall":{"name":"read","args":{"path":"STALE_CALL"}}}]},
        {"role":"user","parts":[{"text":"Start an unrelated task"}]},
        {"role":"user","parts":[{"functionResponse":{"name":"read","response":{"output":"ORPHAN_RESULT_42"}}}]}
    ]});
    let mapped =
        gemini::wrapper::wrap_request(&body, "r28-project", "claude-sonnet-4-5", None, None, None)
            .unwrap();
    let calls = values_at(&mapped, "functionCall");
    let results = values_at(&mapped, "functionResponse");
    for (path, output) in [
        ("EXPLICIT", "EXPLICIT_OUTPUT"),
        ("ANONYMOUS", "ANONYMOUS_OUTPUT"),
    ] {
        let call = calls
            .iter()
            .find(|call| call["args"]["path"] == path)
            .unwrap();
        let result = results
            .iter()
            .find(|result| result["response"]["output"] == output)
            .unwrap();
        assert_eq!(call["id"], result["id"]);
    }
    assert!(!results
        .iter()
        .any(|result| result.to_string().contains("ORPHAN_RESULT_42")));
    assert!(values_at(&mapped, "text").iter().any(|text| text
        .as_str()
        .is_some_and(|text| text.contains("ORPHAN_RESULT_42"))));
}
