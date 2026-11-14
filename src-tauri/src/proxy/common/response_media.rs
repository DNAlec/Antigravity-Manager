//! Visible media output shared by the text-based response adapters.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(crate) type MediaIdentity = (bool, [u8; 32]);

/// Match native Gemini collection: aliases and supplemental metadata do not
/// change the identity of an inline payload or a provider file reference.
pub(crate) fn media_identity(part: &Value) -> MediaIdentity {
    let inline = part.get("inlineData").or_else(|| part.get("inline_data"));
    let file = part.get("fileData").or_else(|| part.get("file_data"));
    let data = inline
        .and_then(|value| value.get("data"))
        .or_else(|| file.and_then(|value| value.get("fileUri").or_else(|| value.get("file_uri"))))
        .and_then(Value::as_str)
        .unwrap_or_default();
    (inline.is_none(), Sha256::digest(data.as_bytes()).into())
}

pub(crate) struct RenderedMedia {
    pub text: String,
    identity: MediaIdentity,
}

pub(crate) fn render_part(part: &Value) -> Option<RenderedMedia> {
    if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data")) {
        return render_inline(
            inline
                .get("mimeType")
                .or_else(|| inline.get("mime_type"))
                .and_then(Value::as_str),
            inline.get("data")?.as_str()?,
        );
    }
    let file = part.get("fileData").or_else(|| part.get("file_data"))?;
    render_file(
        file.get("mimeType")
            .or_else(|| file.get("mime_type"))
            .and_then(Value::as_str),
        file.get("fileUri")
            .or_else(|| file.get("file_uri"))?
            .as_str()?,
    )
}

pub(crate) fn render_inline(mime: Option<&str>, data: &str) -> Option<RenderedMedia> {
    if data.trim().is_empty() {
        return None;
    }
    let mime = mime
        .filter(|mime| !mime.trim().is_empty())
        .unwrap_or("application/octet-stream");
    let target = format!("data:{mime};base64,{data}");
    Some(RenderedMedia {
        text: media_link(mime, &target),
        identity: (false, Sha256::digest(data.as_bytes()).into()),
    })
}

pub(crate) fn render_file(mime: Option<&str>, uri: &str) -> Option<RenderedMedia> {
    if uri.trim().is_empty() {
        return None;
    }
    let text = if url::Url::parse(uri)
        .is_ok_and(|url| matches!(url.scheme(), "http" | "https" | "data"))
    {
        media_link(mime.unwrap_or_default(), uri)
    } else {
        // Provider-local references are useful context, but must not become
        // active Markdown URLs. A variable-width code span also handles ticks.
        let width = uri.split(|ch| ch != '`').map(str::len).max().unwrap_or(0) + 1;
        let fence = "`".repeat(width);
        let reference = uri.replace(['\r', '\n'], " ");
        format!("\n\nFile reference: {fence} {reference} {fence}\n\n")
    };
    Some(RenderedMedia {
        text,
        identity: (true, Sha256::digest(uri.as_bytes()).into()),
    })
}

fn media_link(mime: &str, target: &str) -> String {
    let target = target
        .replace('<', "%3C")
        .replace('>', "%3E")
        .replace('\r', "%0D")
        .replace('\n', "%0A")
        .replace(' ', "%20")
        .replace('\t', "%09");
    let label = if mime.trim().to_ascii_lowercase().starts_with("image/") {
        "![image]"
    } else {
        "[file]"
    };
    format!("\n\n{label}(<{target}>)\n\n")
}

/// Retain the largest multiplicity per candidate, as native collection does.
/// Equal media within one frame are separate outputs; later cumulative copies
/// only update upstream history and must not repeat the public answer.
#[derive(Default)]
pub(crate) struct MediaTracker {
    seen: HashMap<(u64, MediaIdentity), usize>,
    frame: HashMap<(u64, MediaIdentity), usize>,
}

impl MediaTracker {
    pub fn begin_frame(&mut self) {
        self.frame.clear();
    }

    pub fn render(&mut self, candidate: Option<u64>, media: RenderedMedia) -> Option<String> {
        let Some(candidate) = candidate else {
            return Some(media.text);
        };
        let key = (candidate, media.identity);
        let count = self.frame.entry(key).or_default();
        *count += 1;
        let previous = self.seen.entry(key).or_default();
        if *count <= *previous {
            return None;
        }
        *previous = *count;
        Some(media.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn links_keep_images_files_and_provider_references_usable() {
        let image = render_file(Some("image/png"), "https://example.test/a (1)<x>.png")
            .unwrap()
            .text;
        assert!(image.contains("![image](<https://example.test/a%20(1)%3Cx%3E.png>)"));
        for mime in ["application/pdf", "audio/wav", "text/plain"] {
            let text = render_inline(Some(mime), "AQID").unwrap().text;
            assert!(text.contains("[file](<data:"));
            assert!(!text.contains("![image]"));
        }
        for uri in [
            "gs://bucket/result.pdf",
            "javascript:alert(1)",
            "provider://file`name",
        ] {
            let text = render_file(None, uri).unwrap().text;
            assert!(text.contains(uri));
            assert!(!text.contains("]("));
        }
    }

    #[test]
    fn media_multiplicity_matches_native_identity_and_candidate_rules() {
        let original = json!({"inlineData":{"mimeType":"image/png","data":"AQID"}});
        let cumulative = json!({"inline_data":{"data":"AQID","mime_type":"image/png"},"thoughtSignature":"later"});
        assert_eq!(media_identity(&original), media_identity(&cumulative));
        let mut tracker = MediaTracker::default();
        assert!(tracker
            .render(Some(0), render_part(&original).unwrap())
            .is_some());
        assert!(tracker
            .render(Some(0), render_part(&original).unwrap())
            .is_some());
        tracker.begin_frame();
        assert!(tracker
            .render(Some(0), render_part(&cumulative).unwrap())
            .is_none());
        assert!(tracker
            .render(Some(0), render_part(&cumulative).unwrap())
            .is_none());
        assert!(tracker
            .render(Some(0), render_part(&cumulative).unwrap())
            .is_some());
        assert!(tracker
            .render(Some(1), render_part(&cumulative).unwrap())
            .is_some());
        assert!(tracker
            .render(None, render_part(&cumulative).unwrap())
            .is_some());
    }
}
