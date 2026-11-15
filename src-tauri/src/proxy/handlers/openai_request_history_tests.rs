use super::{builtin_history, normalize_responses_plaintext, try_convert_codex_to_openai_request};
use crate::proxy::mappers::{claude, openai};
use serde_json::{json, Value};

const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn texts(value: &Value) -> Vec<&str> {
    match value {
        Value::Object(fields) => fields
            .iter()
            .flat_map(|(key, value)| {
                let mut found = if key == "text" {
                    value.as_str().into_iter().collect()
                } else {
                    Vec::new()
                };
                found.extend(texts(value));
                found
            })
            .collect(),
        Value::Array(values) => values.iter().flat_map(texts).collect(),
        _ => Vec::new(),
    }
}

fn chat(body: Value, responses: bool) -> Value {
    let request: openai::models::OpenAIRequest = serde_json::from_value(body).unwrap();
    super::validate_openai_file_inputs(&request).unwrap();
    openai::request::transform_openai_request_with_session(
        &request,
        "r32-project",
        "gemini-2.5-flash",
        None,
        "r32-readable-history",
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
        "r32-project",
        false,
        None,
        "r32-readable-history",
        None,
    )
    .unwrap()
}

fn responses(items: Vec<Value>) -> Value {
    let mut history = vec![json!({"role":"user","content":"Get the requested data."})];
    history.extend(items);
    history.push(json!({"role":"user","content":"Use the exact values from your prior answer."}));
    json!({"model":"gemini-2.5-flash","input":history})
}

#[test]
fn r32_plain_think_xml_survives_chat_and_claude_request_mapping() {
    for visible in [
        "The exact XML payload is:\n```xml\n<think>INVOICE_XML_42</think>\n```\nCopy it exactly.",
        "Inline `<think>INVOICE_XML_42</think>` is part of the schema.",
        "Business text <think>INVOICE_XML_42</think> remains readable.",
        "<think>INVOICE_XML_42</think> is the exact payload.",
        "<think>INVOICE_XML_42 is an unfinished XML sample.",
    ] {
        let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
            {"role":"user","content":"Give the XML."},
            {"role":"assistant","content":visible},
            {"role":"user","content":"Repeat the exact payload."}
        ]});
        for mapped in [chat(body.clone(), false), anthropic(body)] {
            assert!(texts(&mapped).contains(&visible), "{mapped}");
        }
    }
}

#[test]
fn r32_audio_transcript_preserves_history_order_and_deduplicates_exact_text() {
    let transcript = "Your booking reference is AUDIO_BOOKING_42.";
    for (content, expected) in [
        (Value::Null, json!(transcript)),
        (json!(""), json!(transcript)),
        (json!(transcript), json!(transcript)),
        (
            json!("Readable prefix."),
            json!(format!("Readable prefix.\n{transcript}")),
        ),
        (
            json!([{"type":"text","text":"Readable prefix."}]),
            json!([{"type":"text","text":"Readable prefix."},{"type":"text","text":transcript}]),
        ),
        (
            json!([{"type":"text","text":transcript}]),
            json!([{"type":"text","text":transcript}]),
        ),
    ] {
        let body = json!({"model":"gemini-2.5-flash","messages":[
            {"role":"user","content":"Read the booking reference."},
            {"role":"assistant","content":content,"audio":{"id":"audio_booking","data":"UklGRg==","expires_at":2000000000,"transcript":transcript}},
            {"role":"user","content":"Repeat the spoken reference."}
        ]});
        let typed: openai::models::OpenAIRequest = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(typed).unwrap()["messages"][1]["content"],
            expected
        );
        let mapped = chat(body, false);
        let joined = texts(&mapped).join("\n");
        assert_eq!(joined.matches("AUDIO_BOOKING_42").count(), 1);
        assert!(
            joined.find("AUDIO_BOOKING_42").unwrap()
                < joined.find("Repeat the spoken reference.").unwrap()
        );
    }
    for audio in [
        Value::Null,
        json!({"id":"audio_external"}),
        json!({"transcript":null}),
    ] {
        let body =
            json!({"model":"g","messages":[{"role":"assistant","content":null,"audio":audio}]});
        let typed: openai::models::OpenAIRequest = serde_json::from_value(body).unwrap();
        assert!(serde_json::to_value(typed).unwrap()["messages"][0]["content"].is_null());
    }
    for audio in [
        json!("bad"),
        json!({"transcript":42}),
        json!({"transcript":[]}),
    ] {
        assert!(serde_json::from_value::<openai::models::OpenAIRequest>(
            json!({"model":"g","messages":[{"role":"assistant","content":null,"audio":audio}]})
        )
        .is_err());
    }
}

#[test]
fn r32_responses_builtin_history_retains_logs_errors_shell_and_images_as_readable_content() {
    let body = responses(vec![
        json!({"type":"code_interpreter_call","id":"ci_total","status":"completed","container_id":"cntr_42","code":"print(173)","outputs":[{"type":"logs","logs":"CALCULATED_TOTAL_173"},{"type":"image","url":format!("data:image/png;base64,{PNG}")}]}),
        json!({"type":"mcp_call","id":"mcp_booking","status":"failed","name":"lookup_booking","server_label":"booking","arguments":"{}","output":"MCP_BOOKING_42","error":{"type":"protocol_error","message":"MCP_ERROR_42"}}),
        json!({"type":"shell_call","call_id":"call_shell","action":{"commands":["cat BUILD_INPUT_42"]},"status":"completed"}),
        json!({"type":"shell_call_output","call_id":"call_shell","output":[{"stdout":"SHELL_BUILD_42","stderr":"SHELL_WARNING_42","outcome":{"type":"exit","exit_code":7}},{"stdout":"TIMEOUT_PARTIAL_42","stderr":"","outcome":{"type":"timeout"}}]}),
        json!({"type":"image_generation_call","id":"ig_42","status":"completed","result":PNG,"output_format":"future_optional_format","revised_prompt":"IMAGE_PROMPT_42"}),
    ]);
    let converted = try_convert_codex_to_openai_request(body).unwrap();
    assert!(!converted.to_string().contains("tool_calls"));
    let mapped = chat(converted, true);
    for token in [
        "CALCULATED_TOTAL_173",
        "MCP_BOOKING_42",
        "MCP_ERROR_42",
        "BUILD_INPUT_42",
        "SHELL_BUILD_42",
        "SHELL_WARNING_42",
        "Exit code: 7",
        "TIMEOUT_PARTIAL_42",
        "timed out",
        "IMAGE_PROMPT_42",
        PNG,
    ] {
        assert!(mapped.to_string().contains(token), "{token}: {mapped}");
    }
    assert!(!mapped.to_string().contains("functionCall"));
    assert!(!mapped.to_string().contains("functionResponse"));
}

#[test]
fn r32_responses_local_shell_preserves_official_action_and_output_without_execution() {
    for action in [
        json!({"type":"exec","command":["cat","/tmp/LOCAL_INPUT_42"],"env":{"MODE":"LOCAL_ENV_42"},"working_directory":"/tmp/LOCAL_DIR_42"}),
        json!({"exec":{"command":["cat","/tmp/LOCAL_INPUT_42"],"env":{"MODE":"LOCAL_ENV_42"},"workdir":"/tmp/LOCAL_DIR_42"}}),
        json!({"exec":{"command":"cat /tmp/LOCAL_INPUT_42","env":{"MODE":"LOCAL_ENV_42"},"workdir":"/tmp/LOCAL_DIR_42"}}),
    ] {
        let body = responses(vec![
            json!({"type":"local_shell_call","id":"lsc_42","call_id":"call_local_42","status":"completed","action":action}),
            json!({"type":"local_shell_call_output","id":"call_local_42","status":"completed","output":"{\"stdout\":\"LOCAL_RESULT_42\",\"exit_code\":0}"}),
        ]);
        let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
        for token in [
            "LOCAL_INPUT_42",
            "LOCAL_ENV_42",
            "LOCAL_DIR_42",
            "LOCAL_RESULT_42",
        ] {
            assert!(mapped.to_string().contains(token), "{token}: {mapped}");
        }
        assert!(!mapped.to_string().contains("functionCall"));
    }
}

#[test]
fn r32_builtin_normalization_is_shared_idempotent_and_validates_before_mutation() {
    let body = responses(vec![
        json!({"type":"image_generation_call","id":"img_pending","status":"in_progress","result":null}),
        json!({"type":"code_interpreter_call","id":"ci_partial","status":"incomplete","container_id":"container","code":null,"outputs":[{"type":"logs","logs":"PARTIAL_LOG_42"}]}),
    ]);
    let direct = try_convert_codex_to_openai_request(body.clone()).unwrap();
    let mut http = body;
    normalize_responses_plaintext(&mut http).unwrap();
    let once = http.clone();
    normalize_responses_plaintext(&mut http).unwrap();
    assert_eq!(http, once);
    // The internal ledger intentionally records the representation it received.
    // Both adapter paths must nevertheless send identical conversational content.
    assert_eq!(
        try_convert_codex_to_openai_request(http).unwrap()["messages"],
        direct["messages"]
    );
    let mapped = chat(direct, true);
    for token in ["in_progress", "incomplete", "PARTIAL_LOG_42"] {
        assert!(mapped.to_string().contains(token));
    }
    assert!(!mapped.to_string().contains("completed"));

    for malformed in [
        json!({"type":"local_shell_call","call_id":"lsc","action":{"type":"exec","command":42}}),
        json!({"type":"local_shell_call_output","id":"lsc","output":{}}),
        json!({"type":"mcp_call","id":"mcp","name":"lookup","server_label":"booking","arguments":{}}),
        json!({"type":"code_interpreter_call","id":"ci","container_id":"cntr","outputs":[{"type":"logs","logs":42}]}),
        json!({"type":"shell_call_output","call_id":"sh","output":[{"stdout":"partial","stderr":"","outcome":{"type":"exit"}}]}),
        json!({"type":"image_generation_call","id":"img","result":"invalid-base64"}),
    ] {
        let body = responses(vec![
            json!({"type":"mcp_call","id":"valid","name":"lookup","server_label":"booking","arguments":"{}","output":"VALID_RESULT"}),
            malformed,
        ]);
        let mut normalized = body.clone();
        assert!(normalize_responses_plaintext(&mut normalized).is_err());
        assert_eq!(normalized, body);
        assert!(try_convert_codex_to_openai_request(body).is_err());
    }
    let mut unknown =
        vec![json!({"type":"future_unknown_call","output":"NOT_A_CONFIRMED_TOOL_RESULT"})];
    let before = unknown.clone();
    builtin_history::normalize(&mut unknown).unwrap();
    assert_eq!(unknown, before);
}

#[test]
fn r34_claude_search_named_custom_tools_keep_schema_and_forced_choice() {
    for name in ["web_search", "google_search", "builtin_web_search"] {
        for explicit_type in [false, true] {
            let mut tool = json!({"name":name,"description":"Search a private tenant index","input_schema":{
                "type":"object","properties":{"tenant_id":{"type":"string"}},"required":["tenant_id"]
            }});
            if explicit_type {
                tool["type"] = json!("custom");
            }
            let mapped = anthropic(json!({"model":"gemini-2.5-flash","max_tokens":64,
                "messages":[{"role":"user","content":"Search the private tenant index."}],
                "tools":[tool],"tool_choice":{"type":"tool","name":name}
            }));
            assert_eq!(
                mapped["request"]["tools"][0]["functionDeclarations"][0]["name"],
                name
            );
            assert_eq!(
                mapped["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["required"],
                json!(["tenant_id"])
            );
            assert_eq!(
                mapped["request"]["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
                json!([name])
            );
            assert!(!mapped["request"]["tools"]
                .to_string()
                .contains("googleSearch"));
        }
    }
}

#[test]
fn r34_claude_server_search_and_mixed_client_tools_remain_usable() {
    let body = json!({"model":"gemini-2.5-flash","max_tokens":64,
        "messages":[{"role":"user","content":"Look up the release date."}],
        "tools":[{"type":"web_search_20250305","name":"web_search"}]
    });
    assert_eq!(
        anthropic(body.clone())["request"]["tools"],
        json!([{"googleSearch":{}}])
    );
    let mut mixed = body;
    mixed["tools"][0] = json!({"type":"custom","name":"web_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}}}});
    mixed["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"read","input_schema":{"type":"object","properties":{}}}));
    let mapped = anthropic(mixed);
    let names: Vec<_> = mapped["request"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|tool| tool["functionDeclarations"].as_array())
        .flatten()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["web_search", "read"]);
}

#[test]
fn r34_claude_output_format_keeps_constraints_and_matches_chat_schema_lowering() {
    let schema = json!({"type":"object","properties":{"invoice_code":{"type":"string","enum":["SCHEMA_CODE_42"]}},"required":["invoice_code"],"additionalProperties":false});
    let body = json!({"model":"gemini-2.5-flash","max_tokens":64,
        "messages":[{"role":"user","content":"Return the invoice code."}]
    });
    let mut claude = body.clone();
    claude["output_config"] = json!({"format":{"type":"json_schema","schema":schema}});
    let mut openai = body;
    openai["response_format"] =
        json!({"type":"json_schema","json_schema":{"name":"invoice","schema":schema}});
    let claude = anthropic(claude);
    let openai = chat(openai, false);
    let config = &claude["request"]["generationConfig"];
    assert_eq!(config["responseMimeType"], "application/json");
    assert_eq!(
        config["responseSchema"]["properties"]["invoice_code"]["enum"],
        json!(["SCHEMA_CODE_42"])
    );
    assert_eq!(
        config["responseSchema"],
        openai["request"]["generationConfig"]["responseSchema"]
    );
}

#[test]
fn r34_claude_output_format_schema_errors_reach_the_request_boundary() {
    let request: claude::models::ClaudeRequest = serde_json::from_value(json!({
        "model":"gemini-2.5-flash","max_tokens":64,"messages":[{"role":"user","content":"Return JSON."}],
        "output_config":{"format":{"type":"json_schema","schema":{"type":"string","enum":["x".repeat(crate::proxy::common::json_schema::MAX_SCHEMA_BYTES + 1)]}}}
    })).unwrap();
    assert!(claude::transform_claude_request_in(
        &request,
        "r34-project",
        false,
        None,
        "r34-format-error",
        None
    )
    .is_err());
}

#[test]
fn r34_markdown_source_survives_chat_claude_responses_and_tool_text() {
    let image = format!("![DOC_ALT_42](data:image/png;base64,{PNG})");
    for source in [
        format!("Explain this source exactly:\n```markdown\n{image}\n```"),
        format!("~~~markdown\n{image}\n~~~"),
        format!("> ~~~markdown\n> {image}\n> ~~~"),
        format!("> > ~~~markdown\n> > {image}\n> > ~~~"),
        format!("- example\n  ~~~markdown\n  {image}\n  ~~~"),
        format!("````markdown\n```\n{image}\n```\n````"),
        format!("  ```markdown\n{image}\n  ```"),
        format!("`{image}`"),
        format!("``example ` tick {image}``"),
        format!("`multiline\n{image}\nexample`"),
        format!("\\{image}"),
        format!("    {image}"),
        format!("```markdown\n{image}"),
    ] {
        let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[{"role":"user","content":source}]});
        let mut responses_body = json!({"model":"gemini-2.5-flash","input":[{"role":"user","content":[{"type":"input_text","text":source}]}]});
        normalize_responses_plaintext(&mut responses_body).unwrap();
        let converted = try_convert_codex_to_openai_request(responses_body).unwrap();
        for mapped in [
            chat(body.clone(), false),
            anthropic(body),
            chat(converted, true),
        ] {
            assert!(texts(&mapped).contains(&source.as_str()), "{mapped}");
            assert!(!mapped.to_string().contains("inlineData"), "{mapped}");
        }
        assert_eq!(
            crate::proxy::mappers::common_utils::parse_markdown_images_to_parts(&source),
            vec![json!({"text":source})]
        );
        let mut parts = Vec::new();
        assert_eq!(
            crate::proxy::mappers::common_utils::extract_multimodal_from_tool_text(
                &source, &mut parts
            ),
            source
        );
        assert!(parts.is_empty());
    }
}

#[test]
fn r34_rendered_markdown_and_explicit_images_still_become_visual_input() {
    let image = format!("![VISIBLE_ALT](data:image/png;base64,{PNG})");
    for visible in [
        image.clone(),
        format!("Before {image} after"),
        format!("- screenshot\n    {image}"),
        format!("1. screenshot\n\n    {image}"),
        format!("> - screenshot\n>     {image}"),
        format!("- screenshot\n  - detail\n    {image}"),
        format!("`literal` {image}"),
        format!("\\\\{image}"),
        format!("```\nexample\n```\n{image}"),
        format!("`unmatched {image}"),
        format!("![line one\nline two](data:image/png;base64,{PNG})"),
    ] {
        let parts = crate::proxy::mappers::common_utils::parse_markdown_images_to_parts(&visible);
        assert!(
            parts.iter().any(|part| part["inlineData"]["data"] == PNG),
            "{parts:?}"
        );
        let mut tool_parts = Vec::new();
        crate::proxy::mappers::common_utils::extract_multimodal_from_tool_text(
            &visible,
            &mut tool_parts,
        );
        assert!(
            tool_parts
                .iter()
                .any(|part| part["inlineData"]["data"] == PNG),
            "{tool_parts:?}"
        );
    }
    let mapped = chat(
        json!({"model":"gemini-2.5-flash","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":format!("data:image/png;base64,{PNG}")}}]}]}),
        false,
    );
    assert_eq!(
        mapped["request"]["contents"][0]["parts"][0]["inlineData"]["data"],
        PNG
    );
}

#[test]
fn r34_markdown_code_and_rendered_image_in_same_message_keep_both() {
    let source = format!("`![DOC_ALT_42](data:image/png;base64,{PNG})`");
    let visible = format!("{source}\n![rendered](data:image/png;base64,{PNG})");
    let mapped = chat(
        json!({"model":"gemini-2.5-flash","messages":[{"role":"user","content":visible}]}),
        false,
    );
    assert!(texts(&mapped).join("").contains(&source));
    assert!(mapped["request"]["contents"][0]["parts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|part| part["inlineData"]["data"] == PNG));
}

#[test]
fn r36_completed_file_search_and_patch_history_preserves_results_without_execution() {
    let body = responses(vec![
        json!({"type":"file_search_call","id":"fs_36","queries":["rollback"],"status":"completed","results":[{"filename":"runbook.txt","text":"RETRIEVED_ROLLBACK_36","score":{"future":"optional"}},{"text":"SECOND_RESULT_36"}]}),
        json!({"type":"apply_patch_call","call_id":"patch_36","status":"completed","operation":{"type":"update_file","path":"config.txt","diff":"@@\n-old\n+PATCH_DIFF_36\n"}}),
        json!({"type":"apply_patch_call_output","call_id":"patch_36","status":"failed","output":"PATCH_FAILURE_36: context missing, file unchanged."}),
        json!({"type":"apply_patch_call","call_id":"create_36","status":"completed","operation":{"type":"create_file","path":"created.txt","diff":"+CREATED_BODY_36\n"}}),
        json!({"type":"apply_patch_call","call_id":"delete_36","status":"completed","operation":{"type":"delete_file","path":"deleted.txt"}}),
        json!({"type":"apply_patch_call_output","call_id":"delete_36","status":"completed"}),
    ]);
    let mut normalized = body.clone();
    normalize_responses_plaintext(&mut normalized).unwrap();
    let once = normalized.clone();
    normalize_responses_plaintext(&mut normalized).unwrap();
    assert_eq!(normalized, once);
    let direct = try_convert_codex_to_openai_request(body).unwrap();
    let http = try_convert_codex_to_openai_request(normalized).unwrap();
    assert_eq!(direct["messages"], http["messages"]);
    let mapped = chat(direct, true);
    let contents = mapped["request"]["contents"].to_string();
    for marker in [
        "RETRIEVED_ROLLBACK_36",
        "SECOND_RESULT_36",
        "runbook.txt",
        "PATCH_DIFF_36",
        "PATCH_FAILURE_36",
        "file unchanged",
        "failed",
        "CREATED_BODY_36",
        "created.txt",
        "deleted.txt",
    ] {
        assert!(contents.contains(marker), "{marker}: {contents}");
    }
    assert!(!contents.contains("functionCall"));
    assert!(!contents.contains("functionResponse"));
}

#[test]
fn r36_completed_computer_history_preserves_actions_screenshots_and_unavailable_file_note() {
    for output in [
        json!({"type":"computer_screenshot","image_url":format!("data:image/png;base64,{PNG}")}),
        json!({"type":"computer_screenshot","image_url":"https://example.test/screen.png?signature=KEEP_36"}),
        json!({"type":"computer_screenshot","file_id":"file_SCREEN_36"}),
    ] {
        let body = responses(vec![
            json!({"type":"computer_call","id":"cu_36","call_id":"computer_36","status":"completed","action":{"type":"type","text":"TYPED_VALUE_36"},"pending_safety_checks":[]}),
            json!({"type":"computer_call","id":"cu_batch_36","call_id":"batch_36","status":"completed","actions":[{"type":"click","button":"left","x":42,"y":24},{"type":"type","text":"BATCH_VALUE_36"}]}),
            json!({"type":"computer_call_output","call_id":"computer_36","output":output}),
        ]);
        let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
        let contents = mapped["request"]["contents"].to_string();
        assert!(contents.contains("TYPED_VALUE_36"), "{contents}");
        assert!(contents.contains("BATCH_VALUE_36"), "{contents}");
        if let Some(url) = output["image_url"].as_str() {
            assert!(
                contents.contains(if url.starts_with("data:") { PNG } else { url }),
                "{contents}"
            );
        } else {
            assert!(contents.contains("file_SCREEN_36"), "{contents}");
            assert!(contents.contains("image bytes unavailable"), "{contents}");
        }
        assert!(!contents.contains("functionCall"));
    }
}

#[test]
fn r36_builtin_history_rejects_invalid_content_atomically_and_accepts_absent_optional_results() {
    for malformed in [
        json!({"type":"file_search_call","id":"fs","queries":[1],"results":[]}),
        json!({"type":"file_search_call","id":"fs","queries":[],"results":[{"text":42}]}),
        json!({"type":"apply_patch_call","call_id":"p","operation":{"type":"update_file","path":"x","diff":false}}),
        json!({"type":"apply_patch_call_output","call_id":"p","output":{}}),
        json!({"type":"computer_call","call_id":"c","actions":[42]}),
        json!({"type":"computer_call_output","call_id":"c","output":{"type":"computer_screenshot","image_url":"file:///tmp/private.png"}}),
        json!({"type":"computer_call_output","call_id":"c","output":{"type":"computer_screenshot","image_url":"data:image/png;base64,NOT_BASE64"}}),
    ] {
        let body = responses(vec![
            json!({"type":"file_search_call","id":"valid","queries":[],"results":[{"text":"VALID_RESULT_36"}]}),
            malformed,
        ]);
        let mut normalized = body.clone();
        assert!(normalize_responses_plaintext(&mut normalized).is_err());
        assert_eq!(normalized, body);
        assert!(try_convert_codex_to_openai_request(body).is_err());
    }
    let body = responses(vec![
        json!({"type":"file_search_call","id":"pending","queries":[],"status":"in_progress","results":null}),
        json!({"type":"file_search_call","id":"empty","queries":[],"status":"completed"}),
        json!({"type":"apply_patch_call_output","call_id":"patch","status":"failed","output":null}),
    ]);
    let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
    let contents = mapped["request"]["contents"].to_string();
    assert!(contents.contains("in_progress"));
    assert!(contents.contains("failed"));
}

#[test]
fn r36_text_files_remain_readable_in_chat_responses_and_tool_results() {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let contents = "DOC_BODY_36: 回滚完成.\nPreserve <think>literal source</think>.";
    let data = STANDARD.encode(contents);
    for filename in ["note.txt", "note.md", "note.csv", "note.html"] {
        let body = json!({"model":"gemini-2.5-flash","messages":[{"role":"user","content":[{"type":"text","text":"Read the attachment."},{"type":"file","file":{"filename":filename,"file_data":data}}]}]});
        let response_body = json!({"model":"gemini-2.5-flash","input":[{"role":"user","content":[{"type":"input_text","text":"Read the attachment."},{"type":"input_file","filename":filename,"file_data":data}]}]});
        for mapped in [
            chat(body, false),
            chat(
                try_convert_codex_to_openai_request(response_body).unwrap(),
                true,
            ),
        ] {
            assert!(
                texts(&mapped["request"]["contents"]).contains(&contents),
                "{filename}: {mapped}"
            );
            assert!(!mapped["request"]["contents"]
                .to_string()
                .contains("corrupted"));
        }
    }
    let body = json!({"model":"gemini-2.5-flash","input":[{"role":"user","content":[{"type":"input_file","file_data":format!("data:text/plain;base64,{data}")}]}]});
    let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
    assert!(texts(&mapped["request"]["contents"]).contains(&contents));
    let body = json!({"model":"gemini-2.5-flash","messages":[
        {"role":"user","content":"Read the file."},
        {"role":"assistant","tool_calls":[{"id":"read_36","type":"function","function":{"name":"read","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"read_36","content":[{"type":"file","file":{"filename":"note.txt","file_data":data}}]},
        {"role":"user","content":"Quote the file."}
    ]});
    let mapped = chat(body, false);
    assert!(
        texts(&mapped["request"]["contents"]).contains(&contents),
        "{mapped}"
    );
}

#[test]
fn r38_claude_compaction_keeps_readable_summary_with_optional_metadata() {
    for metadata in [
        json!({}),
        json!({"encrypted_content":"foreign-opaque","signature":"foreign-signature","tool_changes":[]}),
    ] {
        let summary = "COMPACT_38: rollback succeeded at 14:09. Literal <think>example</think>.";
        let mut block = json!({"type":"compaction","content":summary});
        block
            .as_object_mut()
            .unwrap()
            .extend(metadata.as_object().unwrap().clone());
        let mapped = anthropic(
            json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
                {"role":"user","content":"Remember the deployment."},
                {"role":"assistant","content":[{"type":"text","text":"BEFORE_38"},block,{"type":"text","text":"AFTER_38"}]},
                {"role":"user","content":"Repeat the deployment time."}
            ]}),
        );
        assert_eq!(
            texts(&mapped["request"]["contents"]),
            vec![
                "Remember the deployment.",
                "BEFORE_38",
                summary,
                "AFTER_38",
                "Repeat the deployment time."
            ]
        );
        assert!(!mapped["request"]["contents"]
            .to_string()
            .contains("foreign-"));
    }
}

#[test]
fn r38_claude_empty_compaction_is_a_noop_for_ordinary_history() {
    let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
        {"role":"user","content":"Remember the deployment."},
        {"role":"assistant","content":[{"type":"text","text":"READABLE_38: rollback succeeded."}]},
        {"role":"user","content":"Repeat the result."}
    ]});
    let expected = anthropic(body.clone())["request"]["contents"].clone();
    for block in [
        json!({"type":"compaction","content":null}),
        json!({"type":"compaction","content":""}),
        json!({"type":"compaction"}),
    ] {
        let mut input = body.clone();
        input["messages"][1]["content"]
            .as_array_mut()
            .unwrap()
            .insert(0, block.clone());
        assert_eq!(anthropic(input)["request"]["contents"], expected);
        let mut input = body.clone();
        input["messages"]
            .as_array_mut()
            .unwrap()
            .insert(1, json!({"role":"assistant","content":[block]}));
        assert_eq!(anthropic(input)["request"]["contents"], expected);
    }
}

#[test]
fn r38_claude_compaction_rejects_malformed_known_content() {
    for content in [
        json!(17),
        json!({"text":"summary"}),
        json!([{"type":"text","text":"summary"}]),
    ] {
        let body = json!({"model":"gemini-2.5-flash","messages":[{"role":"assistant","content":[{"type":"text","text":"Readable history"},{"type":"compaction","content":content}]}]});
        assert!(serde_json::from_value::<claude::models::ClaudeRequest>(body).is_err());
    }
    assert!(serde_json::from_value::<claude::models::ClaudeRequest>(
        json!({"model":"g","messages":[{"role":"user","content":[{"type":"text","text":17}]}]})
    )
    .is_err());
}

#[test]
fn r38_claude_web_search_errors_remain_readable_without_replaying_tools() {
    for code in [
        "max_uses_exceeded",
        "unavailable",
        "too_many_requests",
        "invalid_tool_input",
        "query_too_long",
        "request_too_large",
    ] {
        let error = json!({"type":"web_search_tool_result_error","error_code":code});
        let mut result = error.clone();
        result["encrypted_content"] = json!("OPAQUE_ERROR_METADATA_38");
        let mapped = anthropic(
            json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
                {"role":"user","content":"Look up the deployment."},
                {"role":"assistant","content":[
                    {"type":"text","text":"SEARCH_START_38"},
                    {"type":"server_tool_use","id":"srvtoolu_38","name":"web_search","input":{"query":"deployment"}},
                    {"type":"web_search_tool_result","tool_use_id":"srvtoolu_38","content":result},
                    {"type":"text","text":"SEARCH_END_38"}
                ]},
                {"role":"user","content":"Why did the lookup fail?"}
            ]}),
        );
        let contents = &mapped["request"]["contents"];
        assert_eq!(
            texts(contents),
            vec![
                "Look up the deployment.",
                "SEARCH_START_38",
                &error.to_string(),
                "SEARCH_END_38",
                "Why did the lookup fail?"
            ]
        );
        assert!(!contents.to_string().contains("functionCall"));
        assert!(!contents.to_string().contains("functionResponse"));
        assert!(!contents.to_string().contains("OPAQUE_ERROR_METADATA_38"));
    }
}

#[test]
fn r38_claude_search_results_keep_titles_and_urls_without_encrypted_bodies() {
    let mapped = anthropic(
        json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
            {"role":"user","content":"Look up the deployment."},
            {"role":"assistant","content":[
                {"type":"server_tool_use","id":"srvtoolu_38","name":"web_search","input":{"query":"deployment"}},
                {"type":"web_search_tool_result","tool_use_id":"srvtoolu_38","content":[
                    {"type":"web_search_result","url":"https://example.com/deployment","title":"Deployment","encrypted_content":"OPAQUE_SEARCH_38"},
                    {"type":"web_search_result","url":"https://example.com/rollback","title":"Rollback","encrypted_content":"OPAQUE_SEARCH_38"},
                    {"type":"future_search_metadata","encrypted_content":"OPAQUE_SEARCH_38"}
                ]},
                {"type":"text","text":"READABLE_SEARCH_38: deployment succeeded."}
            ]},
            {"role":"user","content":"List the page names and links you found."}
        ]}),
    );
    assert_eq!(
        texts(&mapped["request"]["contents"]),
        vec![
            "Look up the deployment.",
            "Search result: Deployment",
            "URL: https://example.com/deployment",
            "Search result: Rollback",
            "URL: https://example.com/rollback",
            "READABLE_SEARCH_38: deployment succeeded.",
            "List the page names and links you found."
        ]
    );
    assert!(!mapped["request"]["contents"]
        .to_string()
        .contains("OPAQUE_SEARCH_38"));
}

fn r38_readable_responses_history(reasoning: Value) -> Value {
    json!({"model":"gemini-2.5-flash","input":[
        {"role":"user","content":[{"type":"input_text","text":"TASK_38: Remember the rollback."}]},
        reasoning,
        {"type":"message","id":"msg_38","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ANSWER_38: rollback succeeded at 14:09.","annotations":[]}]},
        {"role":"user","content":[{"type":"input_text","text":"QUESTION_38: At what time?"},{"type":"input_image","image_url":format!("data:image/png;base64,{PNG}")} ]}
    ]})
}

#[test]
fn r38_responses_foreign_opaque_reasoning_keeps_complete_readable_and_media_history() {
    let opaque = json!({"type":"reasoning","id":"rs_foreign_38","summary":[],"encrypted_content":"foreign-opaque-38","status":"completed"});
    let mut body = r38_readable_responses_history(opaque);
    normalize_responses_plaintext(&mut body).unwrap();
    assert!(!body.to_string().contains("foreign-opaque-38"));
    assert_eq!(body["input"].as_array().unwrap().len(), 3);
    let once = body.clone();
    normalize_responses_plaintext(&mut body).unwrap();
    assert_eq!(body, once);
    let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
    let contents = &mapped["request"]["contents"];
    for marker in ["TASK_38", "ANSWER_38", "QUESTION_38", PNG] {
        assert!(contents.to_string().contains(marker), "{mapped}");
    }
}

#[test]
fn r38_responses_plaintext_reasoning_remains_available_alongside_opaque_metadata() {
    let mut body = r38_readable_responses_history(
        json!({"type":"reasoning","id":"rs_opaque_38","summary":[],"encrypted_content":"foreign-opaque-38"}),
    );
    body["input"].as_array_mut().unwrap().insert(2,json!({"type":"reasoning","id":"rs_readable_38","summary":[{"type":"summary_text","text":"SUMMARY_38: deployment time comes from the recorded result."}],"encrypted_content":"foreign-backup-38"}));
    normalize_responses_plaintext(&mut body).unwrap();
    assert!(!body.to_string().contains("encrypted_content"));
    let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
    let contents = &mapped["request"]["contents"];
    assert!(
        texts(contents).contains(&"SUMMARY_38: deployment time comes from the recorded result.")
    );
    assert!(contents.to_string().contains("ANSWER_38"));
}

#[test]
fn r38_responses_opaque_reasoning_without_complete_readable_history_still_rejects_atomically() {
    let opaque = json!({"type":"reasoning","id":"rs_38","summary":[],"encrypted_content":"foreign-opaque-38"});
    for input in [
        json!([opaque]),
        json!([opaque,{"role":"user","content":"Continue"}]),
        json!([opaque,{"role":"assistant","content":"Only an answer without its task"}]),
        json!([{ "role":"assistant","content":"Answer"},opaque,{"role":"user","content":"Continue"}]),
        json!([{ "role":"user","content":"  "},opaque,{"role":"assistant","content":"Answer"}]),
        json!([{ "role":"user","content":"Task"},opaque,{"role":"assistant","status":"incomplete","content":"Partial"}]),
        json!([{ "role":"user","content":[{"type":"future_unknown","text":"Not readable content"}]},opaque,{"role":"assistant","content":"Answer"}]),
        json!([{ "role":"user","content":"Task"},opaque,{"type":17,"role":"assistant","content":"Malformed message"}]),
        json!([{ "role":"user","content":"Task"},opaque,{"id":"msg_thought_38","role":"assistant","content":"Local transcript-only reasoning"}]),
    ] {
        let mut body = json!({"model":"gemini-2.5-flash","input":input});
        let before = body.clone();
        assert_eq!(
            normalize_responses_plaintext(&mut body).unwrap_err().0,
            "unsupported_reasoning"
        );
        assert_eq!(body, before);
    }
}

#[test]
fn r38_responses_does_not_discard_signed_or_malformed_opaque_state() {
    let opaque = json!({"type":"reasoning","id":"rs_38","summary":[],"encrypted_content":"foreign-opaque-38"});
    for field in ["thoughtSignature", "thought_signature", "signature"] {
        let mut item = opaque.clone();
        item[field] = json!("damaged-native-signature");
        let mut body = r38_readable_responses_history(item);
        let before = body.clone();
        assert_eq!(
            normalize_responses_plaintext(&mut body).unwrap_err().0,
            "unsupported_reasoning"
        );
        assert_eq!(body, before);
    }
    for ciphertext in [
        json!(17),
        json!({"payload":"damaged-state"}),
        json!(["opaque"]),
    ] {
        let mut item = opaque.clone();
        item["encrypted_content"] = ciphertext;
        let mut body = r38_readable_responses_history(item);
        let before = body.clone();
        assert_eq!(
            normalize_responses_plaintext(&mut body).unwrap_err().0,
            "unsupported_reasoning"
        );
        assert_eq!(body, before);
    }
    let mut body = r38_readable_responses_history(
        json!({"type":"compaction","encrypted_content":"unrecoverable"}),
    );
    let before = body.clone();
    assert_eq!(
        normalize_responses_plaintext(&mut body).unwrap_err().0,
        "unsupported_compaction"
    );
    assert_eq!(body, before);
}

#[tokio::test]
async fn r38_responses_websocket_and_prewarm_share_foreign_reasoning_policy() {
    let mut body = r38_readable_responses_history(
        json!({"type":"reasoning","id":"rs_38","summary":[],"encrypted_content":"foreign-opaque-38"}),
    );
    body["type"] = json!("response.create");
    let empty = super::WebsocketSessionState {
        last_request: None,
        last_response_output: json!([]),
        last_response_id: String::new(),
        tool_call_cache: Default::default(),
    };
    let mut socket = empty.clone();
    let normalized =
        super::normalize_responses_websocket_request(body.clone(), &mut socket).unwrap();
    assert!(!normalized.to_string().contains("foreign-opaque-38"));
    let mut prewarm = empty;
    super::prepare_websocket_prewarm(
        &body,
        &mut prewarm,
        &axum::http::HeaderMap::new(),
        "r38-prewarm",
    )
    .await
    .unwrap();
    let stored = prewarm.last_request.as_ref().unwrap();
    assert!(!stored.to_string().contains("foreign-opaque-38"));
    assert_eq!(stored["input"], normalized["input"]);
    let mapped = chat(
        try_convert_codex_to_openai_request(stored.clone()).unwrap(),
        true,
    );
    assert!(mapped["request"]["contents"]
        .to_string()
        .contains("ANSWER_38"));
    assert!(mapped["request"]["contents"].to_string().contains(PNG));

    let invalid = json!({"type":"response.create","model":"gemini-2.5-flash","input":[{"type":"reasoning","summary":[],"encrypted_content":"opaque"},{"role":"user","content":"Continue"}]});
    let before = socket.last_request.clone();
    assert!(super::normalize_responses_websocket_request(invalid.clone(), &mut socket).is_err());
    assert_eq!(socket.last_request, before);
    let before = prewarm.last_request.clone();
    assert!(super::prepare_websocket_prewarm(
        &invalid,
        &mut prewarm,
        &axum::http::HeaderMap::new(),
        "r38-prewarm"
    )
    .await
    .is_err());
    assert_eq!(prewarm.last_request, before);
}

#[tokio::test]
async fn r38_responses_readable_refusal_allows_foreign_metadata_fallback_across_entries() {
    let refusal = "REFUSAL_38: I cannot perform that request, but can discuss alternatives.";
    let mut body = r38_readable_responses_history(
        json!({"type":"reasoning","id":"rs_38","summary":[],"encrypted_content":"foreign-opaque-38"}),
    );
    body["input"][2]["content"] = json!([{"type":"refusal","refusal":refusal}]);
    let mut http = body.clone();
    normalize_responses_plaintext(&mut http).unwrap();
    body["type"] = json!("response.create");
    let empty = super::WebsocketSessionState {
        last_request: None,
        last_response_output: json!([]),
        last_response_id: String::new(),
        tool_call_cache: Default::default(),
    };
    let socket =
        super::normalize_responses_websocket_request(body.clone(), &mut empty.clone()).unwrap();
    let mut prewarm = empty;
    super::prepare_websocket_prewarm(
        &body,
        &mut prewarm,
        &axum::http::HeaderMap::new(),
        "r38-refusal-prewarm",
    )
    .await
    .unwrap();
    for normalized in [http, socket, prewarm.last_request.unwrap()] {
        assert!(!normalized.to_string().contains("foreign-opaque-38"));
        let mapped = chat(
            try_convert_codex_to_openai_request(normalized).unwrap(),
            true,
        );
        let contents = &mapped["request"]["contents"];
        assert!(texts(contents).contains(&refusal), "{mapped}");
        for marker in ["TASK_38", "QUESTION_38", PNG] {
            assert!(contents.to_string().contains(marker), "{mapped}");
        }
    }
}

#[test]
fn r40_claude_fallback_markers_leave_readable_history_unchanged() {
    let body = json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
        {"role":"user","content":"TASK_40: remember the deployment."},
        {"role":"assistant","content":[{"type":"text","text":"BEFORE_40"},{"type":"text","text":"AFTER_40"}]},
        {"role":"user","content":"QUESTION_40: repeat the result."}
    ]});
    let expected = anthropic(body.clone())["request"]["contents"].clone();
    for marker in [
        json!({"type":"fallback","from":{"model":"claude-sonnet-4-6"},"to":{"model":"claude-opus-4-6"},"trigger":null}),
        json!({"type":"fallback","from":{"model":"claude-sonnet-4-6"},"to":{"model":"claude-opus-4-6"},"trigger":{"type":"refusal"},"future_optional":{"opaque":"not conversational content"}}),
        json!({"type":"fallback"}),
    ] {
        let mut mixed = body.clone();
        mixed["messages"][1]["content"]
            .as_array_mut()
            .unwrap()
            .insert(1, marker.clone());
        assert_eq!(anthropic(mixed)["request"]["contents"], expected);
        let mut separate = body.clone();
        separate["messages"]
            .as_array_mut()
            .unwrap()
            .insert(1, json!({"role":"assistant","content":[marker]}));
        assert_eq!(anthropic(separate)["request"]["contents"], expected);
    }
}

fn r40_advisor_history(content: Value) -> Value {
    json!({"model":"gemini-2.5-flash","max_tokens":64,"messages":[
        {"role":"user","content":"TASK_40: inspect the deployment."},
        {"role":"assistant","content":[
            {"type":"text","text":"BEFORE_40"},
            {"type":"server_tool_use","id":"srvtoolu_advisor_40","name":"advisor","input":{"prompt":"Explain the deployment result"}},
            {"type":"advisor_tool_result","tool_use_id":"srvtoolu_advisor_40","content":content},
            {"type":"text","text":"AFTER_40"}
        ]},
        {"role":"user","content":"QUESTION_40: use the actual result."}
    ]})
}

#[test]
fn r40_claude_advisor_results_preserve_literal_text_without_tool_replay() {
    let answer =
        "ADVISOR_40: rollback succeeded. Literal <think>sample</think> and default_api.lookup({}).";
    let body = r40_advisor_history(
        json!({"type":"advisor_result","text":answer,"stop_reason":"end_turn","future_metadata":{"encrypted_content":"OPAQUE_40"}}),
    );
    let mapped = anthropic(body);
    let contents = &mapped["request"]["contents"];
    assert_eq!(
        texts(contents),
        vec![
            "TASK_40: inspect the deployment.",
            "BEFORE_40",
            answer,
            "AFTER_40",
            "QUESTION_40: use the actual result."
        ]
    );
    for forbidden in ["functionCall", "functionResponse", "OPAQUE_40"] {
        assert!(!contents.to_string().contains(forbidden), "{mapped}");
    }
    let empty = anthropic(r40_advisor_history(
        json!({"type":"advisor_result","text":""}),
    ));
    assert!(texts(&empty["request"]["contents"]).contains(&"AFTER_40"));
}

#[test]
fn r40_claude_advisor_errors_and_redacted_results_remain_honest_readable_history() {
    for code in [
        "max_uses_exceeded",
        "prompt_too_long",
        "too_many_requests",
        "overloaded",
        "unavailable",
        "execution_time_exceeded",
        "model_not_found",
    ] {
        let mapped = anthropic(r40_advisor_history(
            json!({"type":"advisor_tool_result_error","error_code":code,"encrypted_content":"OPAQUE_40"}),
        ));
        let contents = &mapped["request"]["contents"];
        assert!(contents.to_string().contains(code), "{mapped}");
        assert!(
            contents.to_string().contains("Advisor tool error"),
            "{mapped}"
        );
        assert!(!contents.to_string().contains("OPAQUE_40"));
        assert!(!contents.to_string().contains("functionCall"));
    }
    let mapped = anthropic(r40_advisor_history(
        json!({"type":"advisor_redacted_result","encrypted_content":"OPAQUE_40","stop_reason":"end_turn"}),
    ));
    let contents = &mapped["request"]["contents"];
    assert!(contents
        .to_string()
        .contains("readable content is unavailable"));
    for marker in ["TASK_40", "BEFORE_40", "AFTER_40", "QUESTION_40"] {
        assert!(contents.to_string().contains(marker));
    }
    assert!(!contents.to_string().contains("OPAQUE_40"));
}

#[test]
fn r40_claude_known_advisor_body_types_still_reject_malformed_values() {
    for content in [
        json!({"type":"advisor_result","text":17}),
        json!({"type":"advisor_result"}),
        json!({"type":"advisor_tool_result_error","error_code":{}}),
        json!({"type":"unknown_result","text":"not an advisor result"}),
    ] {
        assert!(
            serde_json::from_value::<claude::models::ClaudeRequest>(r40_advisor_history(content))
                .is_err()
        );
    }
    let mut invalid = r40_advisor_history(json!({"type":"advisor_result","text":"valid"}));
    invalid["messages"][1]["content"][0] = json!({"type":"text","text":17});
    assert!(serde_json::from_value::<claude::models::ClaudeRequest>(invalid).is_err());
}

fn r40_program_items(status: &str) -> Vec<Value> {
    vec![
        json!({"type":"program","id":"program_40","call_id":"program_call_40","code":"return 'PROGRAM_CODE_40';","fingerprint":"OPAQUE_FINGERPRINT_40"}),
        json!({"type":"program_output","id":"program_result_40","call_id":"program_call_40","status":status,"result":"PROGRAM_RESULT_40: rollback succeeded at 14:09."}),
    ]
}

#[test]
fn r40_responses_program_results_keep_code_body_and_incomplete_status() {
    for status in ["completed", "incomplete"] {
        let mut body = responses(r40_program_items(status));
        normalize_responses_plaintext(&mut body).unwrap();
        let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
        let contents = &mapped["request"]["contents"];
        for marker in [
            "PROGRAM_CODE_40",
            "PROGRAM_RESULT_40",
            status,
            "Use the exact values",
        ] {
            assert!(contents.to_string().contains(marker), "{mapped}");
        }
        for forbidden in ["OPAQUE_FINGERPRINT_40", "functionCall", "functionResponse"] {
            assert!(!contents.to_string().contains(forbidden), "{mapped}");
        }
        if status == "incomplete" {
            assert!(!contents.to_string().contains("Status: completed"));
        }
    }
}

#[test]
fn r40_responses_mcp_tool_listing_preserves_names_descriptions_and_actual_failure() {
    for (tools, error, markers) in [
        (
            json!([]),
            json!("MCP_ERROR_40: server unavailable; no tools discovered."),
            vec!["MCP_ERROR_40"],
        ),
        (
            json!([{"name":"lookup_deployment","input_schema":{"type":"object","properties":{"id":{"type":"string"}}},"description":"MCP_DESCRIPTION_40: look up a deployment.","annotations":{"opaque":"PRIVATE_METADATA_40"}}]),
            Value::Null,
            vec!["lookup_deployment", "MCP_DESCRIPTION_40"],
        ),
        (json!([]), Value::Null, vec!["MCP server: deployments"]),
    ] {
        let mut body = responses(vec![
            json!({"type":"mcp_list_tools","id":"mcp_list_40","server_label":"deployments","tools":tools,"error":error}),
        ]);
        normalize_responses_plaintext(&mut body).unwrap();
        let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
        let contents = &mapped["request"]["contents"];
        for marker in markers {
            assert!(contents.to_string().contains(marker), "{mapped}");
        }
        for forbidden in ["PRIVATE_METADATA_40", "functionCall", "functionResponse"] {
            assert!(!contents.to_string().contains(forbidden), "{mapped}");
        }
    }
}

#[test]
fn r40_responses_optional_tool_metadata_can_degrade_without_rejecting_actual_result() {
    let body = responses(vec![
        json!({"type":"program","id":"p_40","call_id":"cp_40","code":"PROGRAM_CODE_40","fingerprint":{"future":"opaque"}}),
        json!({"type":"program_output","id":"po_40","call_id":"cp_40","status":"incomplete","result":"PROGRAM_RESULT_40","future_optional":[1,2]}),
        json!({"type":"mcp_list_tools","id":"mcp_40","server_label":"deployments","tools":[{"name":"lookup","description":{"future":"metadata"},"input_schema":{"unknown":"schema extension"}}],"error":"MCP_ERROR_40"}),
    ]);
    let mapped = chat(try_convert_codex_to_openai_request(body).unwrap(), true);
    for marker in [
        "PROGRAM_CODE_40",
        "PROGRAM_RESULT_40",
        "MCP_ERROR_40",
        "lookup",
    ] {
        assert!(mapped["request"]["contents"].to_string().contains(marker));
    }
}

#[test]
fn r40_responses_new_builtin_projection_is_shared_and_idempotent() {
    let mut items = r40_program_items("completed");
    items.push(json!({"type":"mcp_list_tools","id":"mcp_40","server_label":"deployments","tools":[],"error":"MCP_ERROR_40"}));
    let body = responses(items);
    let direct = try_convert_codex_to_openai_request(body.clone()).unwrap();
    let mut http = body;
    normalize_responses_plaintext(&mut http).unwrap();
    let once = http.clone();
    normalize_responses_plaintext(&mut http).unwrap();
    assert_eq!(http, once);
    let normalized = try_convert_codex_to_openai_request(http).unwrap();
    assert_eq!(direct["messages"], normalized["messages"]);
    assert_eq!(
        chat(direct, true)["request"]["contents"],
        chat(normalized, true)["request"]["contents"]
    );
}

#[test]
fn r40_responses_malformed_actual_results_fail_before_any_history_is_replaced() {
    for invalid in [
        json!({"type":"program","id":"p","call_id":"cp","code":17}),
        json!({"type":"program_output","id":"po","call_id":"cp","result":{}}),
        json!({"type":"program_output","id":"po","call_id":"cp"}),
        json!({"type":"mcp_list_tools","id":"mcp","server_label":"deployments","tools":[],"error":{}}),
        json!({"type":"mcp_list_tools","id":"mcp","server_label":"deployments","tools":{}}),
        json!({"type":"mcp_list_tools","id":"mcp","server_label":"deployments","tools":[{"name":17}]}),
    ] {
        let mut body = responses(vec![r40_program_items("completed")[1].clone(), invalid]);
        let before = body.clone();
        assert!(normalize_responses_plaintext(&mut body).is_err());
        assert_eq!(body, before);
        assert!(try_convert_codex_to_openai_request(before).is_err());
    }
}

#[tokio::test]
async fn r40_responses_websocket_and_prewarm_keep_program_and_listing_results() {
    let mut items = r40_program_items("incomplete");
    items.push(json!({"type":"mcp_list_tools","id":"mcp_40","server_label":"deployments","tools":[],"error":"MCP_ERROR_40"}));
    let mut body = responses(items);
    body["type"] = json!("response.create");
    let empty = super::WebsocketSessionState {
        last_request: None,
        last_response_output: json!([]),
        last_response_id: String::new(),
        tool_call_cache: Default::default(),
    };
    let mut socket = empty.clone();
    let normalized =
        super::normalize_responses_websocket_request(body.clone(), &mut socket).unwrap();
    let mut prewarm = empty;
    super::prepare_websocket_prewarm(
        &body,
        &mut prewarm,
        &axum::http::HeaderMap::new(),
        "r40-prewarm",
    )
    .await
    .unwrap();
    for normalized in [normalized, prewarm.last_request.clone().unwrap()] {
        let mapped = chat(
            try_convert_codex_to_openai_request(normalized).unwrap(),
            true,
        );
        for marker in ["PROGRAM_RESULT_40", "MCP_ERROR_40", "incomplete"] {
            assert!(
                mapped["request"]["contents"].to_string().contains(marker),
                "{mapped}"
            );
        }
    }
    let mut invalid = body;
    invalid["input"][2]["result"] = json!(17);
    let before = socket.last_request.clone();
    assert!(super::normalize_responses_websocket_request(invalid.clone(), &mut socket).is_err());
    assert_eq!(socket.last_request, before);
    let before = prewarm.last_request.clone();
    assert!(super::prepare_websocket_prewarm(
        &invalid,
        &mut prewarm,
        &axum::http::HeaderMap::new(),
        "r40-prewarm"
    )
    .await
    .is_err());
    assert_eq!(prewarm.last_request, before);
}
