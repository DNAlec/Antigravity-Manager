use super::*;
use crate::proxy::mappers::openai::{collector, streaming};
use crate::proxy::pipeline::stream_error::{observe_response_errors, UpstreamStreamErrorObserver};
use futures::{stream, StreamExt};

#[tokio::test]
async fn r32_responses_json_preserves_missing_finish_as_incomplete() {
    for ending in ["eof", "done", "reset"] {
        let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
        let mut raw = vec![Ok::<_, String>(Bytes::from(format!(
            "data: {}\n\n",
            json!({"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Partial answer"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7}})
        )))];
        if ending == "done" {
            raw.push(Ok(Bytes::from("data: [DONE]\n\n")));
        }
        if ending == "reset" {
            raw.push(Err("connection reset".into()));
        }
        let stream = streaming::create_openai_sse_stream_for_responses_with_selection(
            Box::pin(observe_response_errors(
                stream::iter(raw),
                observer.clone(),
                true,
            )),
            "gemini-flash".into(),
            "r32-json".into(),
            1,
            None,
            true,
            None,
            None,
            observer.candidate_selection().unwrap(),
        );
        let wire: Vec<_> = stream.collect().await;
        let collected = collector::collect_stream_to_json(stream::iter(wire))
            .await
            .unwrap();
        assert!(observer.failure_or_incomplete(None).is_none());
        let response =
            convert_chat_response_to_responses_with_tools(&collected, false, &Default::default());
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "Partial answer"
        );
        assert_eq!(response["usage"]["output_tokens"], 7);
    }
}
