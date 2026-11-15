use super::*;
use crate::proxy::mappers::{gemini, openai::collector};
use futures::stream;

type Wire = Vec<Result<Bytes, String>>;

fn source(frames: Vec<Value>) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> {
    Box::pin(stream::iter(frames.into_iter().map(|frame| {
        Ok(Bytes::from(format!(
            "data: {}\n\n",
            json!({"response":frame})
        )))
    })))
}

fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .filter_map(|chunk| chunk.as_ref().ok())
        .flat_map(|chunk| {
            String::from_utf8_lossy(chunk)
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect::<Vec<Value>>()
        })
        .collect()
}

async fn chat(frames: Vec<Value>) -> Wire {
    create_openai_sse_stream(
        source(frames),
        "gemini-2.5-flash".into(),
        format!("usability-{}", Uuid::new_v4()),
        1,
        None,
        true,
    )
    .collect()
    .await
}

#[tokio::test]
async fn r24_sparse_chat_candidates_keep_text_reasoning_tools_media_and_finish_separate() {
    let wire = chat(vec![
        json!({"candidates":[{"index":1,"content":{"parts":[
            {"thought":true,"text":"THINK_B"},{"text":"B_FIRST"},
            {"functionCall":{"name":"inspect","args":{"path":"src"}}},
            {"fileData":{"mimeType":"image/png","fileUri":"https://example.test/B.png"}}
        ]}}]}),
        json!({"candidates":[{"index":0,"content":{"parts":[
            {"thought":true,"text":"THINK_A"},{"text":"A_ONLY"}
        ]},"finishReason":"STOP"}]}),
        json!({"candidates":[{"index":1,"content":{"parts":[{"text":"B_LAST"}]},"finishReason":"STOP"}]}),
    ]).await;
    let tool_delta = events(&wire)
        .into_iter()
        .find(|event| event.pointer("/choices/0/delta/tool_calls").is_some())
        .unwrap();
    assert_eq!(tool_delta["choices"][0]["index"], 1);
    assert_eq!(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );

    let result = collector::collect_stream_to_json(stream::iter(wire))
        .await
        .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["choices"].as_array().unwrap().len(), 2);
    assert_eq!(result["choices"][0]["index"], 0);
    assert_eq!(result["choices"][0]["message"]["content"], "A_ONLY");
    assert_eq!(
        result["choices"][0]["message"]["reasoning_content"],
        "THINK_A"
    );
    assert!(result["choices"][0]["message"].get("tool_calls").is_none());
    assert_eq!(result["choices"][0]["finish_reason"], "stop");
    let choice = &result["choices"][1];
    assert_eq!(choice["index"], 1);
    assert_eq!(choice["message"]["reasoning_content"], "THINK_B");
    assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);
    assert_eq!(choice["finish_reason"], "tool_calls");
    let text = choice["message"]["content"].as_str().unwrap();
    assert!(
        text.starts_with("B_FIRST") && text.ends_with("B_LAST"),
        "{text}"
    );
    assert!(text.contains("https://example.test/B.png"));
    assert!(!text.contains("A_ONLY"));
}

#[tokio::test]
async fn r24_identical_tools_in_distinct_candidates_have_independent_indices_and_ids() {
    let call = json!({"functionCall":{"name":"inspect","args":{}}});
    let wire = chat(vec![json!({"candidates":[
        {"index":1,"content":{"parts":[call.clone()]},"finishReason":"STOP"},
        {"index":0,"content":{"parts":[call]},"finishReason":"STOP"}
    ]})])
    .await;
    let deltas: Vec<Value> = events(&wire)
        .into_iter()
        .filter(|event| event.pointer("/choices/0/delta/tool_calls").is_some())
        .collect();
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["choices"][0]["index"], 1);
    assert_eq!(deltas[1]["choices"][0]["index"], 0);
    assert_eq!(
        deltas[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );
    assert_eq!(
        deltas[1]["choices"][0]["delta"]["tool_calls"][0]["index"],
        0
    );
    assert_ne!(
        deltas[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
        deltas[1]["choices"][0]["delta"]["tool_calls"][0]["id"]
    );
    let result = collector::collect_stream_to_json(stream::iter(wire))
        .await
        .unwrap();
    assert_eq!(result.choices.len(), 2);
    assert!(result
        .choices
        .iter()
        .all(|choice| choice.message.tool_calls.as_ref().unwrap().len() == 1));
}

#[tokio::test]
async fn r24_collector_reads_every_choice_in_one_sse_event() {
    let event = json!({"choices":[
        {"index":1,"delta":{"content":"B","reasoning_content":"THINK_B"},"finish_reason":"length"},
        {"index":0,"delta":{"content":"A","reasoning_content":"THINK_A"},"finish_reason":"stop"}
    ]});
    let wire: Wire = vec![Ok(Bytes::from(format!("data: {event}\n\n")))];
    let result = collector::collect_stream_to_json(stream::iter(wire))
        .await
        .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(
        result["choices"],
        json!([
            {"index":0,"message":{"role":"assistant","content":"A","reasoning_content":"THINK_A"},"finish_reason":"stop"},
            {"index":1,"message":{"role":"assistant","content":"B","reasoning_content":"THINK_B"},"finish_reason":"length"}
        ])
    );
}

#[tokio::test]
async fn r24_completed_candidate_does_not_suppress_another_candidates_interruption() {
    let input = source(vec![json!({"candidates":[
        {"index":0,"content":{"parts":[{"text":"DONE_A"}]},"finishReason":"STOP"},
        {"index":1,"content":{"parts":[{"text":"PARTIAL_B"}]}},
        {"index":2,"content":{"parts":[{"thought":true,"text":"PARTIAL_REASONING"}]}}
    ]})])
    .chain(stream::iter([Err("connection reset".to_owned())]));
    let wire: Wire = create_openai_sse_stream(
        Box::pin(input),
        "gemini-2.5-flash".into(),
        format!("interrupt-{}", Uuid::new_v4()),
        1,
        None,
        true,
    )
    .collect()
    .await;
    let terminals: Vec<Value> = events(&wire)
        .iter()
        .flat_map(|event| {
            event["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|choice| choice["finish_reason"].is_string())
                .map(|choice| json!([choice["index"], choice["finish_reason"]]))
        })
        .collect();
    assert_eq!(
        terminals,
        vec![
            json!([0, "stop"]),
            json!([1, "length"]),
            json!([2, "length"])
        ]
    );
}

#[tokio::test]
async fn r24_chat_and_responses_preserve_call_multiplicity_and_deduplicate_cumulative_frames() {
    let call = json!({"functionCall":{"name":"inspect","args":{"path":"src"}}});
    let explicit_a =
        json!({"functionCall":{"id":"explicit_a","name":"inspect","args":{"path":"src"}}});
    let explicit_b =
        json!({"functionCall":{"id":"explicit_b","name":"inspect","args":{"path":"src"}}});
    let frame = |parts: Vec<Value>, finish: bool| {
        let mut value = json!({"candidates":[{"index":0,"content":{"parts":parts}}]});
        if finish {
            value["candidates"][0]["finishReason"] = json!("STOP");
        }
        value
    };
    let frames = vec![
        frame(
            vec![
                json!({"text":"TEXT_CONTROL"}),
                json!({"fileData":{"mimeType":"image/png","fileUri":"https://example.test/MEDIA_CONTROL.png"}}),
                call.clone(),
                call.clone(),
                explicit_a.clone(),
                explicit_b.clone(),
            ],
            false,
        ),
        frame(
            vec![
                call.clone(),
                call.clone(),
                explicit_a.clone(),
                explicit_b.clone(),
            ],
            false,
        ),
        frame(
            vec![call.clone(), call.clone(), call, explicit_a, explicit_b],
            true,
        ),
    ];
    let result = collector::collect_stream_to_json(stream::iter(chat(frames.clone()).await))
        .await
        .unwrap();
    let calls = result.choices[0].message.tool_calls.as_ref().unwrap();
    let ids: std::collections::HashSet<&str> = calls.iter().map(|call| call.id.as_str()).collect();
    assert_eq!(calls.len(), 5);
    assert_eq!(ids.len(), 5);
    assert!(ids.contains("explicit_a") && ids.contains("explicit_b"));
    let result = serde_json::to_value(result).unwrap();
    let text = result["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(text.matches("TEXT_CONTROL").count(), 1);
    assert_eq!(text.matches("MEDIA_CONTROL").count(), 1);

    let wire: Wire = create_codex_sse_stream(
        source(frames),
        "gemini-2.5-flash".into(),
        format!("multiplicity-{}", Uuid::new_v4()),
        1,
        1,
        "resp_tools".into(),
        None,
        false,
    )
    .collect()
    .await;
    assert!(wire.iter().all(Result::is_ok));
    let completed = events(&wire)
        .into_iter()
        .find(|event| event["type"] == "response.completed")
        .unwrap();
    let calls: Vec<&Value> = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 5);
    let ids: std::collections::HashSet<&str> = calls
        .iter()
        .map(|call| call["call_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 5);
    assert!(ids.contains("explicit_a") && ids.contains("explicit_b"));
    assert!(completed.to_string().contains("TEXT_CONTROL"));
    assert!(completed.to_string().contains("MEDIA_CONTROL"));
}

fn native(session: &str, buffered: bool) -> gemini::streaming::GeminiStreamHandler {
    gemini::streaming::GeminiStreamHandler::new(
        None,
        session.into(),
        "gemini-2.5-flash".into(),
        1,
        "root".into(),
        false,
        None,
        "test".into(),
        "gemini-2.5-flash".into(),
    )
    .with_candidate_request(1, buffered)
}

#[tokio::test]
async fn r24_native_code_execution_is_deliverable_without_text() {
    for buffered in [false, true] {
        for include_code in [false, true] {
            let session = format!("code-{}", Uuid::new_v4());
            let mut parts = vec![];
            if include_code {
                parts.push(json!({"executableCode":{"language":"PYTHON","code":"print(42)"}}));
            }
            parts.push(json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42"}}));
            let input = json!({"candidates":[{"index":0,"content":{"role":"model","parts":parts},"finishReason":"STOP"}]});
            let wire: Wire = run_stream_lifecycle(
                source(vec![input.clone()]),
                native(&session, buffered),
                StreamLifecycleConfig::new("gemini", "r24", "code".into()),
            )
            .collect()
            .await;
            assert!(wire.iter().all(Result::is_ok), "{wire:?}");
            let result = gemini::collector::collect_stream_to_json(stream::iter(wire), &session)
                .await
                .unwrap();
            assert_eq!(result["candidates"], input["candidates"]);
        }
    }
}

#[tokio::test]
async fn r24_native_code_output_does_not_hide_real_transport_interruption_or_empty_thoughts() {
    let session = format!("code-interrupt-{}", Uuid::new_v4());
    let input = source(vec![json!({"candidates":[{"index":0,"content":{"parts":[
        {"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42"}}
    ]}}]})])
    .chain(stream::iter([Err("connection reset".to_owned())]));
    let wire: Wire = run_stream_lifecycle(
        Box::pin(input),
        native(&session, false),
        StreamLifecycleConfig::new("gemini", "r24", "code".into()),
    )
    .collect()
    .await;
    let frames = events(&wire);
    assert!(frames
        .iter()
        .any(|frame| frame["candidates"][0]["gatewayInterrupted"] == true));
    assert!(!frames
        .iter()
        .any(|frame| frame["candidates"][0]["finishReason"] == "STOP"));

    let wire: Wire = run_stream_lifecycle(
        source(vec![json!({"candidates":[{"index":0,
        "content":{"parts":[{"thought":true,"text":"only thinking"}]},"finishReason":"STOP"}]})]),
        native(&session, false),
        StreamLifecycleConfig::new("gemini", "r24", "thought".into()),
    )
    .collect()
    .await;
    assert!(wire.iter().any(Result::is_err));
}
