//! Readable context from tools that already ran in another Responses session.

use base64::Engine;
use serde_json::{json, Value};

pub(super) fn normalize(items: &mut [Value]) -> Result<(), String> {
    // Validate the complete input before replacing any item in retained history.
    let replacements: Result<Vec<_>, String> = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| match lower(item) {
            Ok(Some(message)) => Some(Ok((index, message))),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect();
    for (index, message) in replacements? {
        items[index] = message;
    }
    Ok(())
}

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("Built-in tool history requires a string {key}"))
}

fn optional_text<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(format!(
            "Built-in tool history {key} must be a string or null"
        )),
    }
}

fn text(parts: &mut Vec<Value>, value: impl Into<String>) {
    let value = value.into();
    if !value.is_empty() {
        parts.push(json!({"type":"input_text","text":value}));
    }
}

fn strings(value: &Value, key: &str) -> Result<String, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty() && values.iter().all(Value::is_string))
        .ok_or_else(|| format!("Built-in tool history requires a non-empty string array {key}"))?;
    Ok(json!(values).to_string())
}

fn image(parts: &mut Vec<Value>, url: &str) -> Result<(), String> {
    if url.starts_with("data:") {
        super::validate_responses_image_data_url(url, 1, 0)?;
    } else {
        let parsed = url::Url::parse(url).map_err(|_| "Invalid tool-result image URL")?;
        if !matches!(parsed.scheme(), "http" | "https" | "gs") || parsed.host_str().is_none() {
            return Err("Invalid tool-result image URL".into());
        }
    }
    parts.push(json!({"type":"input_image","image_url":url}));
    Ok(())
}

fn generated_image(parts: &mut Vec<Value>, result: &str) -> Result<(), String> {
    if result.is_empty() {
        return Ok(());
    }
    if result.len() > super::MAX_INPUT_IMAGE_BYTES.div_ceil(3).saturating_mul(4) {
        return Err("Generated image history exceeds the input image limit".into());
    }
    let data = base64::engine::general_purpose::STANDARD
        .decode(result)
        .map_err(|_| "Generated image history contains invalid base64")?;
    let mime = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if data.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        "image/gif"
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        "image/webp"
    } else if let Ok(svg) = std::str::from_utf8(&data) {
        if svg.trim_start().starts_with("<svg")
            || (svg.trim_start().starts_with("<?xml") && svg.contains("<svg"))
        {
            text(parts, svg);
            return Ok(());
        }
        return Err("Unsupported generated image history format".into());
    } else {
        return Err("Unsupported generated image history format".into());
    };
    // Detect the actual bytes; optional output_format need not be understood.
    image(parts, &format!("data:{mime};base64,{result}"))
}

fn lower(item: &Value) -> Result<Option<Value>, String> {
    let kind = item["type"].as_str().unwrap_or_default();
    if !matches!(
        kind,
        "code_interpreter_call"
            | "mcp_call"
            | "mcp_list_tools"
            | "program"
            | "program_output"
            | "shell_call"
            | "shell_call_output"
            | "image_generation_call"
            | "local_shell_call"
            | "local_shell_call_output"
            | "file_search_call"
            | "apply_patch_call"
            | "apply_patch_call_output"
            | "computer_call"
            | "computer_call_output"
    ) {
        return Ok(None);
    }
    let mut parts = Vec::new();
    let id_key = if matches!(
        kind,
        "shell_call"
            | "shell_call_output"
            | "local_shell_call"
            | "apply_patch_call"
            | "apply_patch_call_output"
            | "computer_call"
            | "computer_call_output"
            | "program"
            | "program_output"
    ) {
        "call_id"
    } else {
        "id"
    };
    let call_id = required_text(item, id_key)?;
    if call_id.trim().is_empty() {
        return Err("Built-in tool history is missing its call identity".into());
    }
    text(&mut parts, format!("Tool history {kind} (call {call_id}):"));
    if let Some(status) = optional_text(item, "status")? {
        text(&mut parts, format!("Status: {status}"));
    }
    match kind {
        "program" => {
            text(
                &mut parts,
                format!("Code:\n{}", required_text(item, "code")?),
            );
        }
        "program_output" => text(&mut parts, required_text(item, "result")?),
        "mcp_list_tools" => {
            let server = required_text(item, "server_label")?;
            text(&mut parts, format!("MCP server: {server}"));
            let tools = item
                .get("tools")
                .and_then(Value::as_array)
                .ok_or("MCP listing history requires a tools array")?;
            for tool in tools {
                let name = required_text(tool, "name")?;
                text(&mut parts, format!("Available tool: {server}/{name}"));
                if let Some(description) = tool.get("description").and_then(Value::as_str) {
                    text(&mut parts, description);
                }
            }
            if let Some(error) = optional_text(item, "error")? {
                text(&mut parts, format!("MCP tool listing error: {error}"));
            }
        }
        "file_search_call" => {
            if let Some(queries) = item.get("queries") {
                if !queries
                    .as_array()
                    .is_some_and(|queries| queries.iter().all(Value::is_string))
                {
                    return Err("File search queries must be a string array".into());
                }
                text(&mut parts, format!("Queries: {queries}"));
            }
            match item.get("results") {
                None | Some(Value::Null) => {}
                Some(Value::Array(results)) => {
                    for result in results {
                        if !result.is_object() {
                            return Err("File search result must be an object".into());
                        }
                        if let Some(filename) = result.get("filename").and_then(Value::as_str) {
                            text(&mut parts, format!("File: {filename}"));
                        }
                        if let Some(body) = optional_text(result, "text")? {
                            text(&mut parts, body);
                        }
                    }
                }
                _ => return Err("File search results must be an array or null".into()),
            }
        }
        "apply_patch_call" => {
            let operation = item
                .get("operation")
                .filter(|value| value.is_object())
                .ok_or("Patch history is missing its operation")?;
            let kind = required_text(operation, "type")?;
            let path = required_text(operation, "path")?;
            text(&mut parts, format!("Patch operation: {kind}\nPath: {path}"));
            if matches!(kind, "create_file" | "update_file") {
                text(
                    &mut parts,
                    format!("Diff:\n{}", required_text(operation, "diff")?),
                );
            } else if kind != "delete_file" {
                // Future operation details remain readable without executing them.
                text(&mut parts, operation.to_string());
            }
        }
        "apply_patch_call_output" => {
            if let Some(output) = optional_text(item, "output")? {
                text(&mut parts, output);
            }
        }
        "computer_call" => {
            if let Some(action) = item.get("action").filter(|value| !value.is_null()) {
                if !action.is_object() {
                    return Err("Computer action must be an object".into());
                }
                text(&mut parts, format!("Computer action: {action}"));
            }
            if let Some(actions) = item.get("actions").filter(|value| !value.is_null()) {
                if !actions
                    .as_array()
                    .is_some_and(|actions| actions.iter().all(Value::is_object))
                {
                    return Err("Computer actions must be an object array".into());
                }
                text(&mut parts, format!("Computer actions: {actions}"));
            }
        }
        "computer_call_output" => {
            let output = item
                .get("output")
                .filter(|value| value.is_object())
                .ok_or("Computer output history requires a screenshot object")?;
            if output["type"] != "computer_screenshot" {
                return Err("Unknown computer output content".into());
            }
            if let Some(url) = optional_text(output, "image_url")? {
                image(&mut parts, url)?;
            } else {
                // A provider-owned file ID cannot be resolved by this gateway.
                // Keep that limitation visible rather than dropping the history.
                let file_id = required_text(output, "file_id")?;
                text(
                    &mut parts,
                    format!(
                        "Screenshot stored as external file {file_id}; image bytes unavailable."
                    ),
                );
            }
        }
        "code_interpreter_call" => {
            required_text(item, "container_id")?;
            if let Some(code) = optional_text(item, "code")? {
                text(&mut parts, format!("Code:\n{code}"));
            }
            match item.get("outputs") {
                None | Some(Value::Null) => {}
                Some(Value::Array(outputs)) => {
                    for output in outputs {
                        match output["type"].as_str() {
                            Some("logs") => text(&mut parts, required_text(output, "logs")?),
                            Some("image") => image(&mut parts, required_text(output, "url")?)?,
                            _ => return Err("Unknown code interpreter output content".into()),
                        }
                    }
                }
                _ => return Err("Code interpreter outputs must be an array or null".into()),
            }
        }
        "mcp_call" => {
            let server = required_text(item, "server_label")?;
            let name = required_text(item, "name")?;
            let arguments = required_text(item, "arguments")?;
            text(
                &mut parts,
                format!("MCP tool: {server}/{name}\nArguments: {arguments}"),
            );
            if let Some(output) = optional_text(item, "output")? {
                text(&mut parts, output);
            }
            if let Some(error) = item.get("error").filter(|value| !value.is_null()) {
                if !error.is_object() {
                    return Err("MCP history error must be an object".into());
                }
                text(&mut parts, format!("Tool error: {error}"));
            }
        }
        "shell_call" | "local_shell_call" => {
            let action = item
                .get("action")
                .filter(|value| value.is_object())
                .ok_or("Shell history is missing its action")?;
            let action = if kind == "local_shell_call" {
                if let Some(exec) = action.get("exec") {
                    if !exec.is_object() {
                        return Err("Local shell exec must be an object".into());
                    }
                    exec
                } else {
                    if action["type"] != "exec" {
                        return Err("Unknown local shell action".into());
                    }
                    action
                }
            } else {
                action
            };
            let key = if kind == "local_shell_call" {
                "command"
            } else {
                "commands"
            };
            let command = if kind == "local_shell_call" && item["action"].get("exec").is_some() {
                match action.get(key).and_then(Value::as_str) {
                    Some(command) => json!([command]).to_string(),
                    None => strings(action, key)?,
                }
            } else {
                strings(action, key)?
            };
            text(&mut parts, format!("Command: {command}"));
            if let Some(env) = action.get("env").filter(|value| !value.is_null()) {
                if !env
                    .as_object()
                    .is_some_and(|env| env.values().all(Value::is_string))
                {
                    return Err("Shell environment must map names to strings".into());
                }
                text(&mut parts, format!("Environment: {env}"));
            }
            for key in ["working_directory", "workdir", "user"] {
                if let Some(value) = optional_text(action, key)? {
                    text(&mut parts, format!("{key}: {value}"));
                }
            }
        }
        "shell_call_output" => {
            let outputs = item
                .get("output")
                .and_then(Value::as_array)
                .ok_or("Shell output history requires an output array")?;
            for output in outputs {
                let stdout = required_text(output, "stdout")?;
                let stderr = required_text(output, "stderr")?;
                if !stdout.is_empty() {
                    text(&mut parts, format!("stdout:\n{stdout}"));
                }
                if !stderr.is_empty() {
                    text(&mut parts, format!("stderr:\n{stderr}"));
                }
                match output["outcome"]["type"].as_str() {
                    Some("exit") => {
                        let code = output["outcome"]["exit_code"]
                            .as_i64()
                            .ok_or("Shell exit outcome requires exit_code")?;
                        text(&mut parts, format!("Exit code: {code}"));
                    }
                    Some("timeout") => text(&mut parts, "Shell command timed out."),
                    _ => return Err("Unknown shell output outcome".into()),
                }
            }
        }
        "local_shell_call_output" => text(&mut parts, required_text(item, "output")?),
        "image_generation_call" => {
            if let Some(prompt) = optional_text(item, "revised_prompt")? {
                text(&mut parts, prompt);
            }
            if let Some(result) = optional_text(item, "result")? {
                generated_image(&mut parts, result)?;
            }
            if let Some(error) = item.get("error").filter(|value| !value.is_null()) {
                if !error.is_object() {
                    return Err("Image history error must be an object".into());
                }
                text(&mut parts, format!("Image generation error: {error}"));
            }
        }
        _ => return Ok(None),
    }
    let mut message = json!({"type":"message","role":"assistant","content":parts});
    if let Some(id) = item.get("id") {
        if !id.is_null() && !id.is_string() {
            return Err("Tool history id must be a string".into());
        }
        message["id"] = id.clone();
    }
    Ok(Some(message))
}
