use super::*;
use crate::proxy::mappers::{claude, gemini, openai};
use crate::proxy::pipeline::stream_error::{observe_response_errors, UpstreamStreamErrorObserver};
use futures::stream;
use std::io;

type Wire = Vec<Result<Bytes, String>>;

fn candidate(index: u64, text: &str, finish: Option<&str>) -> Value {
    let mut value = json!({"index":index,"content":{"role":"model","parts":[{"text":text}]}});
    if let Some(reason) = finish {
        value["finishReason"] = json!(reason);
    }
    value
}

fn frame(candidates: Vec<Value>) -> Result<Bytes, String> {
    let value = json!({"response":{"candidates":candidates,"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7}}});
    Ok(Bytes::from(format!("data: {value}\n\n")))
}

fn source(
    raw: Wire,
    observer: &UpstreamStreamErrorObserver,
    collecting: bool,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> {
    Box::pin(observe_response_errors(
        stream::iter(raw),
        observer.clone(),
        collecting,
    ))
}

fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .flat_map(|chunk| {
            String::from_utf8_lossy(chunk.as_ref().expect("rendered SSE"))
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str(data).ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Claude,
    Responses,
    Intermediate,
}
const ROUTES: [Route; 3] = [Route::Claude, Route::Responses, Route::Intermediate];

struct Output {
    events: Vec<Value>,
    collected: Result<Value, String>,
    observer: UpstreamStreamErrorObserver,
}

impl Output {
    fn text(&self, route: Route) -> String {
        self.events
            .iter()
            .filter_map(|event| match route {
                Route::Claude => event.pointer("/delta/text").and_then(Value::as_str),
                Route::Responses => (event["type"] == "response.output_text.delta")
                    .then(|| event["delta"].as_str())
                    .flatten(),
                Route::Intermediate => event
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str),
            })
            .collect()
    }
}

async fn render(raw: Wire, route: Route) -> Output {
    render_with_collection(raw, route, true).await
}

async fn render_with_collection(raw: Wire, route: Route, collecting: bool) -> Output {
    let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
    let selection = observer.candidate_selection().unwrap();
    let sid = format!("r30-selection-{}", Uuid::new_v4());
    let wire: Wire = match route {
        Route::Claude => {
            claude::create_claude_sse_stream_with_selection(
                source(raw, &observer, collecting),
                "r30".into(),
                "r30@example.test".into(),
                None,
                false,
                200_000,
                None,
                1,
                None,
                vec![],
                None,
                selection,
            )
            .collect()
            .await
        }
        Route::Responses => {
            create_codex_sse_stream_with_selection(
                source(raw, &observer, collecting),
                "gemini-2.5-flash".into(),
                sid.clone(),
                1,
                1,
                sid,
                None,
                false,
                None,
                Default::default(),
                selection,
            )
            .collect()
            .await
        }
        Route::Intermediate => {
            create_openai_sse_stream_for_responses_with_selection(
                source(raw, &observer, collecting),
                "gemini-2.5-flash".into(),
                sid,
                1,
                None,
                true,
                None,
                None,
                selection,
            )
            .collect()
            .await
        }
    };
    let events = events(&wire);
    let collected = match route {
        Route::Claude => claude::collect_stream_to_json(stream::iter(
            wire.into_iter()
                .map(|chunk| chunk.map_err(io::Error::other)),
        ))
        .await
        .map(|value| serde_json::to_value(value).unwrap()),
        Route::Intermediate => openai::collector::collect_stream_to_json(stream::iter(wire))
            .await
            .map(|value| serde_json::to_value(value).unwrap()),
        Route::Responses => {
            let terminal = events.last().expect("Responses terminal");
            if terminal["type"] == "response.completed" {
                Ok(terminal["response"].clone())
            } else {
                Err(json!({"error":terminal["response"]["error"],"partial_response":terminal["response"]}).to_string())
            }
        }
    };
    Output {
        events,
        collected,
        observer,
    }
}

struct Case {
    name: &'static str,
    raw: Wire,
    a: &'static str,
    b: &'static str,
    selected_failure: bool,
    any_failure: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "stable-order-control",
            raw: vec![
                frame(vec![
                    candidate(0, "A_BEGIN_", None),
                    candidate(1, "B_BEGIN_", None),
                ]),
                frame(vec![
                    candidate(0, "A_END", Some("STOP")),
                    candidate(1, "B_END", Some("STOP")),
                ]),
            ],
            a: "A_BEGIN_A_END",
            b: "B_BEGIN_B_END",
            selected_failure: false,
            any_failure: false,
        },
        Case {
            name: "reordered-two-candidates",
            raw: vec![
                frame(vec![
                    candidate(0, "A_BEGIN_", None),
                    candidate(1, "B_BEGIN_", None),
                ]),
                frame(vec![
                    candidate(1, "B_END", Some("STOP")),
                    candidate(0, "A_END", Some("STOP")),
                ]),
            ],
            a: "A_BEGIN_A_END",
            b: "B_BEGIN_B_END",
            selected_failure: false,
            any_failure: false,
        },
        Case {
            name: "interleaved-singleton-candidates",
            raw: vec![
                frame(vec![candidate(0, "A_BEGIN_", None)]),
                frame(vec![candidate(1, "B_COMPLETE", Some("STOP"))]),
                frame(vec![candidate(0, "A_END", Some("STOP"))]),
            ],
            a: "A_BEGIN_A_END",
            b: "B_COMPLETE",
            selected_failure: false,
            any_failure: false,
        },
        Case {
            name: "unselected-failure-becomes-first",
            raw: vec![
                frame(vec![
                    candidate(0, "A_BEGIN_", None),
                    candidate(1, "B_BEGIN_", None),
                ]),
                frame(vec![
                    candidate(1, "B_FAILED", Some("MALFORMED_FUNCTION_CALL")),
                    candidate(0, "A_END", Some("STOP")),
                ]),
            ],
            a: "A_BEGIN_A_END",
            b: "B_BEGIN_B_FAILED",
            selected_failure: false,
            any_failure: true,
        },
        Case {
            name: "selected-failure-becomes-second",
            raw: vec![
                frame(vec![
                    candidate(0, "A_BEGIN_", None),
                    candidate(1, "B_BEGIN_", None),
                ]),
                frame(vec![
                    candidate(1, "B_END", Some("STOP")),
                    candidate(0, "A_FAILED", Some("MALFORMED_FUNCTION_CALL")),
                ]),
            ],
            a: "A_BEGIN_A_FAILED",
            b: "B_BEGIN_B_END",
            selected_failure: true,
            any_failure: true,
        },
        Case {
            name: "later-unselected-failure-control",
            raw: vec![
                frame(vec![candidate(0, "A_COMPLETE", Some("STOP"))]),
                frame(vec![candidate(
                    1,
                    "B_FAILED",
                    Some("MALFORMED_FUNCTION_CALL"),
                )]),
            ],
            a: "A_COMPLETE",
            b: "B_FAILED",
            selected_failure: false,
            any_failure: true,
        },
    ]
}

#[tokio::test]
async fn r30_selected_candidate_identity_survives_reordering_and_sparse_frames() {
    for case in cases() {
        for route in ROUTES {
            let output = render(case.raw.clone(), route).await;
            assert_eq!(output.text(route), case.a, "{} {route:?}", case.name);
            assert_eq!(
                output.collected.is_err(),
                case.selected_failure,
                "{} {route:?}",
                case.name
            );
            let failure = output
                .observer
                .failure_or_incomplete(output.collected.as_ref().err().map(String::as_str));
            assert_eq!(
                failure.is_some(),
                case.selected_failure,
                "{} {route:?}",
                case.name
            );
            assert_eq!(
                output.observer.candidate_selection().unwrap().index(),
                Some(0)
            );
            if let Some(failure) = failure {
                assert_eq!(failure.status, 502);
                assert!(!failure.replay_safe());
                assert!(failure.body.contains(case.a), "{} {route:?}", case.name);
                assert!(!failure.body.contains("B_BEGIN_"));
                assert!(!failure.body.contains("B_END"));
            } else {
                let collected = output.collected.unwrap();
                assert!(
                    !collected.to_string().contains("B_"),
                    "{} {route:?}",
                    case.name
                );
                if matches!(route, Route::Intermediate) {
                    assert_eq!(collected["choices"].as_array().unwrap().len(), 1);
                    assert_eq!(collected["choices"][0]["message"]["content"], case.a);
                }
            }
        }
    }
}

#[tokio::test]
async fn r30_chat_and_native_gemini_keep_all_candidate_identities() {
    for case in cases() {
        let observer = UpstreamStreamErrorObserver::default();
        let chat: Wire = create_openai_sse_stream(
            source(case.raw.clone(), &observer, true),
            "gemini-2.5-flash".into(),
            format!("r30-chat-{}", case.name),
            1,
            None,
            true,
        )
        .collect()
        .await;
        let collected = openai::collector::collect_stream_to_json(stream::iter(chat)).await;
        assert_eq!(collected.is_err(), case.any_failure, "{}", case.name);
        assert_eq!(observer.failure().is_some(), case.any_failure);
        let body = match collected {
            Ok(body) => serde_json::to_value(body).unwrap(),
            Err(error) => {
                serde_json::from_str::<Value>(&error).unwrap()["partial_response"].clone()
            }
        };
        for (index, expected) in [(0, case.a), (1, case.b)] {
            let choice = body["choices"]
                .as_array()
                .unwrap()
                .iter()
                .find(|choice| choice["index"] == index)
                .unwrap();
            assert_eq!(choice["message"]["content"], expected, "{} Chat", case.name);
        }
        let observer = UpstreamStreamErrorObserver::default();
        let sid = format!("r30-native-{}", case.name);
        let handler = gemini::streaming::GeminiStreamHandler::new(
            None,
            sid.clone(),
            "gemini-2.5-flash".into(),
            1,
            "root".into(),
            false,
            None,
            "r30".into(),
            "gemini-2.5-flash".into(),
        )
        .with_candidate_request(2, true);
        let native = run_stream_lifecycle(
            source(case.raw, &observer, true),
            handler,
            StreamLifecycleConfig::new("gemini", "r30", case.name.into()),
        );
        let collected = gemini::collector::collect_stream_to_json(native, &sid)
            .await
            .unwrap();
        assert_eq!(observer.failure().is_some(), case.any_failure);
        for (index, expected) in [(0, case.a), (1, case.b)] {
            let candidate = collected["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|candidate| candidate["index"] == index)
                .unwrap();
            let text: String = candidate["content"]["parts"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect();
            assert_eq!(text, expected, "{} Gemini", case.name);
        }
    }
}

#[tokio::test]
async fn r30_selected_tools_media_and_finish_share_first_nonzero_identity() {
    let mut selected = candidate(7, "SELECTED_BEGIN_", None);
    selected["content"]["parts"].as_array_mut().unwrap().push(json!({"functionCall":{"id":"call_selected","name":"selected_tool","args":{"q":"selected"}}}));
    selected["content"]["parts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"inlineData":{"mimeType":"image/png","data":"U0VMRUNURUQ="}}));
    let mut other = candidate(0, "UNSELECTED_TEXT", Some("MALFORMED_FUNCTION_CALL"));
    other["content"]["parts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"functionCall":{"id":"call_other","name":"other_tool","args":{"q":"other"}}}));
    other["content"]["parts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"inlineData":{"mimeType":"image/png","data":"T1RIRVI="}}));
    let raw = vec![
        frame(vec![selected]),
        frame(vec![other.clone()]),
        frame(vec![other, candidate(7, "SELECTED_END", Some("STOP"))]),
    ];
    for route in ROUTES {
        let output = render(raw.clone(), route).await;
        let text = output.text(route);
        assert!(text.starts_with("SELECTED_BEGIN_"), "{route:?}");
        assert!(text.contains("U0VMRUNURUQ="), "{route:?}");
        assert!(text.ends_with("SELECTED_END"), "{route:?}");
        let wire = serde_json::to_string(&output.events).unwrap();
        assert!(wire.contains("selected_tool"), "{route:?}");
        assert!(!wire.contains("other_tool"), "{route:?}");
        assert!(!wire.contains("T1RIRVI="), "{route:?}");
        assert!(!wire.contains("UNSELECTED_TEXT"), "{route:?}");
        assert!(output.collected.is_ok(), "{route:?}");
        assert!(output.observer.failure().is_none(), "{route:?}");
        assert_eq!(
            output.observer.candidate_selection().unwrap().index(),
            Some(7)
        );
    }
}

#[tokio::test]
async fn r30_large_first_frame_uses_renderer_selection_even_when_observer_skips_it() {
    let big_text = "A".repeat(270_000);
    for selected_failure in [false, true] {
        let first = frame(vec![candidate(7, &big_text, None)]).unwrap();
        let other = frame(vec![candidate(
            0,
            "UNSELECTED_FAILED",
            Some("MALFORMED_FUNCTION_CALL"),
        )])
        .unwrap();
        let finish = frame(vec![candidate(
            7,
            "SELECTED_END",
            Some(if selected_failure {
                "MALFORMED_FUNCTION_CALL"
            } else {
                "STOP"
            }),
        )])
        .unwrap();
        // One network read lets the bounded observer see all frames before the renderer pins 7.
        let bytes = [first.as_ref(), other.as_ref(), finish.as_ref()].concat();
        let bounded = UpstreamStreamErrorObserver::default().with_selected_candidate();
        let _: Wire = source(vec![Ok(Bytes::from(bytes.clone()))], &bounded, false)
            .collect()
            .await;
        // Without the full-frame renderer, the bounded observer first sees index 0.
        assert!(bounded
            .failure()
            .unwrap()
            .body
            .contains("UNSELECTED_FAILED"));
        for collecting in [false, true] {
            for route in ROUTES {
                let output =
                    render_with_collection(vec![Ok(Bytes::from(bytes.clone()))], route, collecting)
                        .await;
                assert_eq!(
                    output.text(route),
                    format!("{big_text}SELECTED_END"),
                    "{route:?} collecting={collecting}"
                );
                assert_eq!(
                    output.collected.is_err(),
                    selected_failure,
                    "{route:?} collecting={collecting}"
                );
                assert_eq!(
                    output.observer.failure().is_some(),
                    selected_failure,
                    "{route:?} collecting={collecting}"
                );
                assert_eq!(
                    output.observer.candidate_selection().unwrap().index(),
                    Some(7)
                );
            }
        }
    }
}

#[tokio::test]
async fn r30_global_structured_error_still_fails_selected_routes() {
    for route in ROUTES {
        let output = render(
            vec![
                frame(vec![candidate(7, "SELECTED_PARTIAL", None)]),
                Ok(Bytes::from(
                    "data: {\"error\":{\"code\":503,\"message\":\"overloaded\"}}\n\n",
                )),
            ],
            route,
        )
        .await;
        assert_eq!(output.text(route), "SELECTED_PARTIAL");
        assert!(output.collected.is_err(), "{route:?}");
        assert_eq!(output.observer.failure().unwrap().status, 503);
    }
}
