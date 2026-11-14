use serde_json::{json, Value};

pub const MAX_CLEAN_RECURSION_DEPTH: usize = 64;
pub const MAX_REF_RECURSION_DEPTH: usize = 64;
pub const MAX_DESCRIPTION_LENGTH: usize = 8192;

// Bound both traversal work and allocation, including descriptions and enum data.
// Depth alone cannot constrain a shared, acyclic reference graph.
pub const MAX_SCHEMA_NODES: usize = 32_768;
pub const MAX_SCHEMA_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct SchemaBudget {
    nodes: usize,
    bytes: usize,
}

impl SchemaBudget {
    fn charge(&mut self, value: &Value) -> Result<(), String> {
        self.charge_at_depth(value, 0)
    }

    fn charge_size(&mut self, nodes: usize, bytes: usize) -> Result<(), String> {
        self.nodes = self.nodes.saturating_add(nodes);
        self.bytes = self.bytes.saturating_add(bytes);
        if self.nodes > MAX_SCHEMA_NODES || self.bytes > MAX_SCHEMA_BYTES {
            return Err("schema expansion exceeds the supported resource budget".into());
        }
        Ok(())
    }

    fn clone_value(&mut self, value: &Value) -> Result<Value, String> {
        self.charge(value)?;
        Ok(value.clone())
    }

    fn enum_string(&mut self, value: &Value) -> Result<String, String> {
        // JSON escaping can enlarge enum objects. Bound the output as it is
        // written, before growing the buffer, rather than after to_string().
        self.charge_size(1, 0)?;
        let mut writer = SchemaJsonWriter {
            budget: self,
            bytes: Vec::new(),
        };
        serde_json::to_writer(&mut writer, value).map_err(|e| e.to_string())?;
        String::from_utf8(writer.bytes).map_err(|e| e.to_string())
    }

    fn charge_at_depth(&mut self, value: &Value, depth: usize) -> Result<(), String> {
        self.charge_size(
            1,
            match value {
                Value::String(s) => s.len(),
                _ => 16,
            },
        )?;
        if depth > 256 {
            return Err("schema expansion exceeds the supported resource budget".into());
        }
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    self.charge_size(0, key.len())?;
                    self.charge_at_depth(child, depth + 1)?;
                }
            }
            Value::Array(items) => {
                for child in items {
                    self.charge_at_depth(child, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

struct SchemaJsonWriter<'a> {
    budget: &'a mut SchemaBudget,
    bytes: Vec<u8>,
}

impl std::io::Write for SchemaJsonWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.budget
            .charge_size(0, bytes.len())
            .map_err(std::io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// 规范化工具或参数的描述文本：
/// 1. 保持客户端原始换行与排版格式不变，避免折叠导致指令可读性下降；
/// 2. 仅对超过安全预算（8192 字符）的极端超长描述进行截断，防止上游解析溢出或拒收。
pub fn sanitize_description(s: &str) -> String {
    if s.chars().count() <= MAX_DESCRIPTION_LENGTH {
        s.to_string()
    } else {
        let truncated: String = s
            .chars()
            .take(MAX_DESCRIPTION_LENGTH.saturating_sub(15))
            .collect();
        format!("{}... [truncated]", truncated)
    }
}

/// 递归清理 JSON Schema 以符合 Gemini 接口要求
///
/// 1. [New] 展开 $ref 和 $defs: 将引用替换为实际定义，解决 Gemini 不支持 $ref 的问题
/// 2. 移除不支持的字段: $schema, additionalProperties, format, default, uniqueItems, validation fields
/// 3. 处理联合类型: ["string", "null"] -> "string"
/// 4. [NEW] 处理 anyOf 联合类型: anyOf: [{"type": "string"}, {"type": "null"}] -> "type": "string"
/// 5. 将 type 字段的值转换为小写 (Gemini v1internal 要求)
/// 6. 移除数字校验字段: multipleOf, exclusiveMinimum, exclusiveMaximum 等
/// 清洗用于 responseSchema 的 JSON Schema
pub fn clean_response_schema(value: &mut Value) -> Result<(), String> {
    clean_json_schema(value)
}

pub fn clean_json_schema(value: &mut Value) -> Result<(), String> {
    let mut budget = SchemaBudget::default();
    budget.charge(value)?;
    // Commit only after successful expansion; callers never see a partial schema.
    let mut expanded = value.clone();
    clean_schema_bounded(&mut expanded, value, &mut budget)?;
    *value = expanded;
    Ok(())
}

fn clean_schema_bounded(
    value: &mut Value,
    root: &Value,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    // Borrow definitions from the unchanged input. Exact JSON Pointers take
    // precedence over the legacy lookup for clients with nested definitions.
    let mut defs = std::collections::HashMap::new();
    collect_all_defs(root, &mut defs);
    if let Value::Object(map) = value {
        flatten_refs(map, root, &defs, 0, &mut Default::default(), budget)?;
    }
    clean_json_schema_recursive(value, true, 0, budget)?;
    budget.charge(value)
}

/// 清洗 JSON Schema 以符合 Gemini 接口要求
///
/// 遵循纯粹协议透传原则，不执行任何特定工具的私有适配逻辑
pub fn clean_json_schema_for_tool(value: &mut Value, _tool_name: &str) -> Result<(), String> {
    clean_json_schema(value)
}

/// Check that best-effort conversion stays within the resource budget.
/// Schema fidelity differences are handled by normalization, not request rejection.
pub fn validate_tool_schema(schema: &Value) -> Result<(), String> {
    let mut budget = SchemaBudget::default();
    let mut normalized = budget.clone_value(schema)?;
    clean_schema_bounded(&mut normalized, schema, &mut budget)
}

fn collect_all_defs<'a>(value: &'a Value, defs: &mut std::collections::HashMap<String, &'a Value>) {
    if let Value::Object(map) = value {
        for key in ["$defs", "definitions"] {
            if let Some(children) = map.get(key).and_then(Value::as_object) {
                for (name, schema) in children {
                    defs.entry(name.clone()).or_insert(schema);
                }
            }
        }
        for (key, child) in map {
            if key != "$defs" && key != "definitions" {
                collect_all_defs(child, defs);
            }
        }
    } else if let Value::Array(items) = value {
        for child in items {
            collect_all_defs(child, defs);
        }
    }
}

fn reference_fallback(
    map: &mut serde_json::Map<String, Value>,
    target: Option<&Value>,
    reference: &str,
    cyclic: bool,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    // Keep known structural information at the expansion boundary. Do not
    // invent tool arguments or recursively copy the referenced children.
    let kind = target
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .unwrap_or(if cyclic { "object" } else { "string" });
    if !map.contains_key("properties") && !map.contains_key("items") {
        map.entry("type".to_string()).or_insert_with(|| json!(kind));
    }
    let prefix = if cyclic { "Cyclic" } else { "Unresolved" };
    budget.charge_size(1, reference.len().saturating_add(32))?;
    let hint = format!("({prefix} $ref: {reference})");
    let description = map
        .entry("description".to_string())
        .or_insert_with(|| json!(""));
    if let Value::String(text) = description {
        if !text.contains(&hint) {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&hint);
        }
    }
    Ok(())
}

fn flatten_refs(
    map: &mut serde_json::Map<String, Value>,
    root: &Value,
    defs: &std::collections::HashMap<String, &Value>,
    depth: usize,
    active_refs: &mut std::collections::HashSet<String>,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    budget.charge_size(1, 16)?;
    map.remove("$defs");
    map.remove("definitions");
    if let Some(Value::String(reference)) = map.remove("$ref") {
        let target = reference
            .strip_prefix('#')
            .and_then(|pointer| root.pointer(pointer))
            .or_else(|| {
                // Older clients place named definitions below the root but
                // still refer to them with root-level definition paths.
                reference
                    .strip_prefix("#/$defs/")
                    .or_else(|| reference.strip_prefix("#/definitions/"))
                    .filter(|name| !name.contains('/'))
                    .and_then(|name| defs.get(name).copied())
            });
        let cyclic = active_refs.contains(&reference);
        if cyclic || depth >= MAX_REF_RECURSION_DEPTH {
            reference_fallback(map, target, &reference, cyclic, budget)?;
        } else if let Some(target) = target.filter(|target| target.is_object()) {
            if let Value::Object(mut resolved) = budget.clone_value(target)? {
                active_refs.insert(reference.clone());
                flatten_refs(&mut resolved, root, defs, depth + 1, active_refs, budget)?;
                active_refs.remove(&reference);
                merge_all_of(&mut resolved, budget)?;
                merge_schema_constraints(map, resolved, 0, budget)?;
            }
        } else {
            reference_fallback(map, target, &reference, false, budget)?;
        }
    }
    if depth > MAX_REF_RECURSION_DEPTH {
        return Ok(());
    }
    // Visit schema edges only; property names and enum values are business data.
    if let Some(children) = map.get_mut("properties").and_then(Value::as_object_mut) {
        for child in children.values_mut().filter_map(Value::as_object_mut) {
            flatten_refs(child, root, defs, depth + 1, active_refs, budget)?;
        }
    }
    if let Some(items) = map.get_mut("items").and_then(Value::as_object_mut) {
        flatten_refs(items, root, defs, depth + 1, active_refs, budget)?;
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(items) = map.get_mut(key).and_then(Value::as_array_mut) {
            for child in items.iter_mut().filter_map(Value::as_object_mut) {
                flatten_refs(child, root, defs, depth + 1, active_refs, budget)?;
            }
        }
    }
    Ok(())
}

/// [FIX #3682] 递归深度达到上限时的防御性合规兜底（Fail-safe Fallback）。
///
/// 使用基于堆的显式迭代遍历，杜绝调用栈溢出风险。
/// 确保即便是超过最大递归深度的深层节点，也不会将违背 Gemini Schema Protobuf 约束的非法结构透传上游：
/// 1. 所有 type == "array" 的节点若缺失 items，强制注入 items: {"type": "string"}；
/// 2. 包含 properties 的节点若缺失 type，补全 type: "object"；
/// 3. 若 properties 不是 Object，归一化为 {}；
/// 4. 若 type == "object" 或包含 properties 却包含 items，移除 items 避免 Gemini 400；
/// 5. 剔除非 Object 的 items，防止 bare boolean 等非法项上送。
fn apply_minimal_failsafe(root: &mut Value, budget: &mut SchemaBudget) -> Result<(), String> {
    let mut stack = vec![root];
    while let Some(current) = stack.pop() {
        budget.charge_size(1, 16)?;
        match current {
            Value::Object(map) => {
                let is_array = match map.get("type") {
                    Some(Value::String(s)) => s.eq_ignore_ascii_case("array"),
                    Some(Value::Array(arr)) => arr.iter().any(|v| {
                        v.as_str()
                            .map(|s| s.eq_ignore_ascii_case("array"))
                            .unwrap_or(false)
                    }),
                    _ => false,
                };
                let is_object = if is_array {
                    false
                } else {
                    map.get("type")
                        .and_then(Value::as_str)
                        .map(|t| t.eq_ignore_ascii_case("object"))
                        .unwrap_or(false)
                        || map.contains_key("properties")
                };

                if is_array {
                    // [FIX #3682] 数组节点不可携带 properties（Gemini Protobuf 严格禁止 ARRAY 包含 properties）
                    // 并且必须保障 items 存在，杜绝因包含 properties 误走 is_object 将 items 删除导致 400 INVALID_ARGUMENT
                    map.insert("type".to_string(), Value::String("array".to_string()));
                    map.remove("properties");
                    if let Some(items) = map.get("items") {
                        if !items.is_object() {
                            map.remove("items");
                        }
                    }
                    if !map.contains_key("items") {
                        map.insert("items".to_string(), json!({ "type": "string" }));
                    }
                } else if is_object {
                    if !map.contains_key("type") && !map.contains_key("anyOf") {
                        map.insert("type".to_string(), Value::String("object".to_string()));
                    }
                    map.remove("items");

                    if let Some(props) = map.get_mut("properties") {
                        if !props.is_object() {
                            *props = json!({});
                        }
                    }
                }

                // 压入子节点继续迭代合规兜底
                for v in map.values_mut() {
                    if v.is_object() || v.is_array() {
                        stack.push(v);
                    }
                }
            }
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    if item.is_object() || item.is_array() {
                        stack.push(item);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn clean_json_schema_recursive(
    value: &mut Value,
    is_schema_node: bool,
    depth: usize,
    budget: &mut SchemaBudget,
) -> Result<bool, String> {
    budget.charge_size(1, 16)?;
    if depth > MAX_CLEAN_RECURSION_DEPTH {
        tracing::warn!(
            depth,
            max = MAX_CLEAN_RECURSION_DEPTH,
            "[Schema-Clean] Max recursion depth reached in clean_json_schema_recursive, applying minimal failsafe"
        );
        apply_minimal_failsafe(value, budget)?;
        return Ok(false);
    }
    let mut is_effectively_nullable = false;

    match value {
        Value::Object(map) => {
            // Resolve compositions before inferring a type from properties/items.
            // allOf can introduce a type array; normalize unions before healing
            // their object/array structure.
            merge_all_of(map, budget)?;
            // Gemini has anyOf, not oneOf. Relax exclusivity rather than
            // rejecting otherwise usable object alternatives.
            if let Some(branches) = map.remove("oneOf") {
                map.entry("anyOf".to_string()).or_insert(branches);
            }
            if map.contains_key("anyOf") && map.get("type").is_some_and(Value::is_array) {
                // Keep the explicit alternatives instead of selecting the first
                // type or materializing a potentially exponential intersection.
                map.remove("type");
            }
            if map.get("type").is_some_and(Value::is_array) {
                // The type list is already owned by this schema; move it instead
                // of duplicating it before building the union alternatives.
                if !map.contains_key("anyOf") && !map.contains_key("oneOf") {
                    // JSON Schema applies these keywords only to their own
                    // types. Move them into the matching alternatives before
                    // array defaults or object healing can change their meaning.
                    let constraints: Vec<_> = [
                        ("items", "array"),
                        ("minItems", "array"),
                        ("maxItems", "array"),
                        ("properties", "object"),
                        ("required", "object"),
                        ("minProperties", "object"),
                        ("maxProperties", "object"),
                        ("minLength", "string"),
                        ("maxLength", "string"),
                        ("pattern", "string"),
                        ("minimum", "number"),
                        ("maximum", "number"),
                    ]
                    .into_iter()
                    .filter_map(|(key, kind)| map.remove(key).map(|value| (key, kind, value)))
                    .collect();
                    if let Some(Value::Array(types)) = map.remove("type") {
                        let mut branches = Vec::new();
                        for kind in types {
                            budget.charge_size(1, 16)?;
                            let kind_name = kind.as_str().unwrap_or_default().to_ascii_lowercase();
                            let mut branch = serde_json::Map::new();
                            branch.insert("type".into(), kind);
                            for (key, target, constraint) in &constraints {
                                if kind_name == *target
                                    || (kind_name == "integer" && *target == "number")
                                {
                                    branch.insert((*key).into(), budget.clone_value(constraint)?);
                                }
                            }
                            branches.push(Value::Object(branch));
                        }
                        map.insert("anyOf".into(), Value::Array(branches));
                    }
                }
            }

            // 0.1 [NEW #3327] 规范化 const 关键字 (转换为 enum 与对应 type)
            // Gemini/Vertex 的 Schema proto 不支持 const，直接传入会导致 400 INVALID_ARGUMENT
            // 例如 {"const": "element"} -> {"type": "string", "enum": ["element"]}
            if let Some(const_val) = map.remove("const") {
                if !map.contains_key("type") && !map.contains_key("anyOf") {
                    let inferred_type = match &const_val {
                        Value::String(_) => Some("string"),
                        Value::Number(n) => {
                            if n.is_i64() || n.is_u64() {
                                Some("integer")
                            } else {
                                Some("number")
                            }
                        }
                        Value::Bool(_) => Some("boolean"),
                        Value::Array(_) => Some("array"),
                        Value::Object(_) => Some("object"),
                        Value::Null => None,
                    };
                    if let Some(t) = inferred_type {
                        map.insert("type".to_string(), Value::String(t.to_string()));
                    }
                }
                let enum_entry = map
                    .entry("enum".to_string())
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Value::Array(enum_arr) = enum_entry {
                    if !enum_arr.contains(&const_val) {
                        enum_arr.push(const_val);
                    }
                }
            }

            // 0.5 [NEW] 结构归一化 (Normalization)
            // 针对某些 MCP 工具（如 pencil）误用 items 定义对象属性的情况进行修复。
            // 如果 type=object 或包含 properties，但又定义了 items，Gemini 会因为 items 只能出现在 array 中而报错。
            // 我们将 items 的内容“对齐”到 properties 中。
            // [FIX #3682] 注意：若节点显式为 array 类型，严禁误判为 Object 并偷走 items，而是应该移除其非法携带的 properties！
            let is_explicit_array = map
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t.eq_ignore_ascii_case("array"))
                .unwrap_or(false);

            if is_explicit_array {
                map.remove("properties");
            } else if !map.contains_key("anyOf")
                && (map.get("type").and_then(|t| t.as_str()) == Some("object")
                    || map.contains_key("properties"))
            {
                if let Some(items) = map.remove("items") {
                    tracing::warn!("[Schema-Normalization] Found 'items' in an Object-like node. Moving content to 'properties'.");
                    let target_props = map
                        .entry("properties".to_string())
                        .or_insert_with(|| json!({}));
                    if let Some(target_map) = target_props.as_object_mut() {
                        if let Value::Object(source_map) = items {
                            for (k, v) in source_map {
                                target_map.entry(k).or_insert(v);
                            }
                        }
                    }
                }
            }

            // 1. [CRITICAL] 深度递归处理子项
            // 处理 properties (对象)
            // [FIX] Gemini's Schema proto requires `properties` to be an object (map<string, Schema>).
            // Non-object values (null, [], boolean, etc.) trigger upstream 400 errors.
            // Normalize non-object `properties` to an empty object `{}`.
            if let Some(props_val) = map.get_mut("properties") {
                if !props_val.is_object() {
                    *props_val = json!({});
                }
            }

            if let Some(Value::Object(props)) = map.get_mut("properties") {
                // [FIX] Drop boolean / non-object sub-schemas. JSON Schema allows
                // `prop: true|false`, but Gemini's Schema proto requires every property
                // value to be an object; a bare boolean triggers an upstream 400
                // ("Invalid value at '...properties[N].value' ... false").
                let dropped_keys: Vec<String> = props
                    .iter()
                    .filter(|(_, v)| !v.is_object())
                    .map(|(k, _)| k.clone())
                    .collect();
                for k in &dropped_keys {
                    props.remove(k);
                }

                for v in props.values_mut() {
                    // properties 的每一个值都必须是一个独立的 Schema 节点
                    clean_json_schema_recursive(v, true, depth + 1, budget)?;
                }
                let valid_keys: std::collections::HashSet<String> = props.keys().cloned().collect();

                if let Some(Value::Array(req_arr)) = map.get_mut("required") {
                    req_arr.retain(|r| r.as_str().map(|s| valid_keys.contains(s)).unwrap_or(false));
                    if req_arr.is_empty() {
                        map.remove("required");
                    }
                }

                // [NEW] 隐式类型注入：如果有 properties 但没 type，补全为 object
                if !map.contains_key("type") && !map.contains_key("anyOf") {
                    map.insert("type".to_string(), Value::String("object".to_string()));
                }
            }

            // 处理 items (数组)
            // [FIX] items must be a Schema object; drop bare boolean / invalid items
            // (JSON Schema allows boolean `items`, Gemini's Schema proto rejects it).
            if map.get("items").map(|i| !i.is_object()).unwrap_or(false) {
                map.remove("items");
            }
            if let Some(items) = map.get_mut("items") {
                // items 的内容必须是一个独立的 Schema 节点
                clean_json_schema_recursive(items, true, depth + 1, budget)?;

                // [NEW] 隐式类型注入：如果有 items 但没 type，补全为 array
                if !map.contains_key("type") && !map.contains_key("anyOf") {
                    map.insert("type".to_string(), Value::String("array".to_string()));
                }
            }

            // Gemini's Schema proto requires every ARRAY node to declare `items`,
            // including nested arrays. JSON Schema permits an itemless array, so
            // clients such as Claude Code may legitimately emit {"type":"array"}.
            // Use a string item schema as a Gemini-compatible fallback for these
            // otherwise unconstrained arrays.
            let is_array = match map.get("type") {
                Some(Value::String(s)) => s.eq_ignore_ascii_case("array"),
                Some(Value::Array(arr)) => arr.iter().any(|v| {
                    v.as_str()
                        .map(|s| s.eq_ignore_ascii_case("array"))
                        .unwrap_or(false)
                }),
                _ => false,
            };
            if is_array && !map.contains_key("items") {
                map.insert("items".to_string(), json!({ "type": "string" }));
            }

            // Fallback: 对既没有 properties 也没有 items 的常规对象进行清理
            if !map.contains_key("properties") && !map.contains_key("items") {
                for (k, v) in map.iter_mut() {
                    // 排除掉关键字
                    if k != "anyOf" && k != "oneOf" && k != "allOf" && k != "enum" && k != "type" {
                        clean_json_schema_recursive(v, false, depth + 1, budget)?;
                    }
                }
            }

            // Gemini Schema supports anyOf. Selecting a single branch silently
            // changes the tool contract (for example, integer|string -> string).
            let can_inline_union = map.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "anyOf" | "title" | "description" | "default" | "$schema"
                )
            });
            let parent_items = if map.get("anyOf").is_some_and(Value::is_array) {
                map.get("items")
                    .map(|items| budget.clone_value(items))
                    .transpose()?
            } else {
                None
            };
            if let Some(Value::Array(branches)) = map.get_mut("anyOf") {
                let is_null = |v: &Value| {
                    v.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|t| t.eq_ignore_ascii_case("null"))
                        && v.as_object().is_some_and(|o| o.len() == 1)
                };
                let had_null = can_inline_union
                    && branches.iter().any(|v| !is_null(v))
                    && branches.iter().any(|v| {
                        v.get("type")
                            .and_then(Value::as_str)
                            .is_some_and(|t| t.eq_ignore_ascii_case("null"))
                            && v.as_object().is_some_and(|o| o.len() == 1)
                    });
                if had_null {
                    branches.retain(|v| {
                        !(v.get("type")
                            .and_then(Value::as_str)
                            .is_some_and(|t| t.eq_ignore_ascii_case("null"))
                            && v.as_object().is_some_and(|o| o.len() == 1))
                    });
                }
                for branch in branches {
                    if let (Some(items), Some(branch_map)) =
                        (parent_items.as_ref(), branch.as_object_mut())
                    {
                        merge_all_of(branch_map, budget)?;
                        let accepts_array = match branch_map.get("type") {
                            Some(Value::String(kind)) => kind.eq_ignore_ascii_case("array"),
                            Some(Value::Array(types)) => types.iter().any(|kind| {
                                kind.as_str()
                                    .is_some_and(|s| s.eq_ignore_ascii_case("array"))
                            }),
                            _ => {
                                branch_map.contains_key("anyOf") || branch_map.contains_key("oneOf")
                            }
                        };
                        if accepts_array {
                            // Parent and explicit branch constraints remain a
                            // conjunction. Inherit only a missing item schema,
                            // preventing the default string items from adding
                            // an unrelated restriction to this alternative.
                            if !branch_map.contains_key("items") {
                                branch_map.insert("items".into(), budget.clone_value(items)?);
                            }
                        }
                    }
                    clean_json_schema_recursive(branch, true, depth + 1, budget)?;
                }
                if had_null {
                    map.insert("nullable".to_string(), json!(true));
                }
                if can_inline_union
                    && map
                        .get("anyOf")
                        .and_then(Value::as_array)
                        .is_some_and(|a| a.len() == 1)
                {
                    if let Some(Value::Array(mut branches)) = map.remove("anyOf") {
                        if let Some(Value::Object(branch)) = branches.pop() {
                            for (key, value) in branch {
                                map.entry(key).or_insert(value);
                            }
                        }
                    }
                }
            }
            // 3. [SAFETY] 检查当前对象是否为 JSON Schema 节点
            // 只有当对象看起来像 Schema (包含 type, properties, items, enum, anyOf 等) 时，才执行白名单过滤。
            // 否则，如果它是一个普通的 Value (如 request.rs 中的 functionCall 对象)，直接应用激进过滤会破坏结构。
            let allowed_fields = [
                "type",
                "description",
                "properties",
                "required",
                "items",
                "enum",
                "title",
                "anyOf",
                "nullable",
                "minItems",
                "maxItems",
                "minLength",
                "maxLength",
                "minimum",
                "maximum",
                "pattern",
                "minProperties",
                "maxProperties",
            ];

            let has_standard_keyword = map.keys().any(|k| allowed_fields.contains(&k.as_str()));

            // [NEW] 启发式修复：如果明确是 Schema 节点，但没有标准关键字，却有其他 Key
            // 我们推测这是一个“简写”的对象定义，尝试将其内部 Key 移动到 properties 中。
            // 补充：必须确保它不是工具调用或结果 (含有 functionCall/functionResponse/args 等)，防止运行时参数结构被破坏。
            let is_not_schema_payload = map.contains_key("functionCall")
                || map.contains_key("functionResponse")
                || map.contains_key("args")
                || map.contains_key("tool_calls")
                || map.contains_key("tool_call_id");
            if is_schema_node && !has_standard_keyword && !map.is_empty() && !is_not_schema_payload
            {
                let mut properties = serde_json::Map::new();
                let keys: Vec<String> = map.keys().cloned().collect();
                for k in keys {
                    if let Some(v) = map.remove(&k) {
                        properties.insert(k, v);
                    }
                }
                map.insert("type".to_string(), Value::String("object".to_string()));
                map.insert("properties".to_string(), Value::Object(properties));

                // 递归清理刚刚移动进去的属性
                if let Some(Value::Object(props_map)) = map.get_mut("properties") {
                    for v in props_map.values_mut() {
                        clean_json_schema_recursive(v, true, depth + 1, budget)?;
                    }
                }
            }

            let looks_like_schema = is_schema_node && !is_not_schema_payload;

            if looks_like_schema {
                // 4. [CRITICAL] 白名单过滤：彻底物理移除 Gemini 不支持的内容，防止 400 错误
                let keys_to_remove: Vec<String> = map
                    .keys()
                    .filter(|k| !allowed_fields.contains(&k.as_str()))
                    .cloned()
                    .collect();
                for k in keys_to_remove {
                    map.remove(&k);
                }

                // 5. 保持客户端原始 description 文本，不折叠换行，不截断内容，保证完全保真透传

                // 6. [SAFETY] 处理空 Object
                // [FIX] 移除 reason 字段注入逻辑
                // 之前的实现会为空 Object 注入 reason 字段，导致 Gemini CLI 等工具报 "malformed function call"
                // 因为模型会生成包含 reason 参数的调用，但工具定义中并没有这个参数
                // 现在改为：空 Object 保持空的 properties，让 Gemini 模型自行决定是否需要参数
                let is_object_type = map
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t.eq_ignore_ascii_case("object"))
                    .unwrap_or(false);
                if is_object_type {
                    if !map.contains_key("properties") {
                        map.insert("properties".to_string(), serde_json::json!({}));
                    }
                }

                // 7. [SAFETY] Required 字段对齐
                let valid_prop_keys: Option<std::collections::HashSet<String>> = map
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .map(|obj| obj.keys().cloned().collect());

                if let Some(required_val) = map.get_mut("required") {
                    if let Some(req_arr) = required_val.as_array_mut() {
                        if let Some(keys) = &valid_prop_keys {
                            req_arr
                                .retain(|k| k.as_str().map(|s| keys.contains(s)).unwrap_or(false));
                        }
                    }
                }

                if !map.contains_key("type") && !map.contains_key("anyOf") {
                    if map.contains_key("enum") {
                        map.insert("type".to_string(), Value::String("string".to_string()));
                    } else if map.contains_key("properties") {
                        map.insert("type".to_string(), Value::String("object".to_string()));
                    } else if map.contains_key("items") {
                        map.insert("type".to_string(), Value::String("array".to_string()));
                    } else if ["pattern", "minLength", "maxLength"]
                        .iter()
                        .any(|key| map.contains_key(*key))
                    {
                        map.insert("type".to_string(), json!("string"));
                    } else if ["minimum", "maximum"]
                        .iter()
                        .any(|key| map.contains_key(*key))
                    {
                        map.insert("type".to_string(), json!("number"));
                    }
                }

                // [IMPROVED] 提前计算回退类型以避免借用冲突
                let fallback = if map.contains_key("properties") {
                    "object"
                } else if map.contains_key("items") {
                    "array"
                } else {
                    "string"
                };

                // 8. 处理 type 字段
                if let Some(type_val) = map.get_mut("type") {
                    let mut selected_type = None;
                    match type_val {
                        Value::String(s) => {
                            let lower = s.to_lowercase();
                            selected_type = Some(lower);
                        }
                        Value::Array(arr) => {
                            for item in arr {
                                if let Value::String(s) = item {
                                    let lower = s.to_lowercase();
                                    if lower == "null" {
                                        is_effectively_nullable = true;
                                    } else if selected_type.is_none() {
                                        selected_type = Some(lower);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    *type_val =
                        Value::String(selected_type.unwrap_or_else(|| fallback.to_string()));
                }

                if is_effectively_nullable {
                    let desc_val = map
                        .entry("description".to_string())
                        .or_insert_with(|| Value::String("".to_string()));
                    if let Value::String(s) = desc_val {
                        if !s.contains("nullable") {
                            if !s.is_empty() {
                                s.push(' ');
                            }
                            s.push_str("(nullable)");
                        }
                    }
                }

                // 9. [FIX #2041] Enum 强制规范化为符合 Gemini Protobuf 约束的形式：
                // a. Enum 元素强制转字符串，并剔除空字符串 (Gemini 约束: Schema.enum[i]: cannot be empty)
                // b. 若剔除后数组为空，则移除 enum 属性
                // c. Gemini Protobuf 约束 Schema.enum: only allowed for STRING type，因此包含有效 enum 的字段其 type 必须归一化为 string
                let mut has_valid_enum = false;
                if let Some(enum_val) = map.get_mut("enum") {
                    if let Value::Array(arr) = enum_val {
                        for item in arr.iter_mut() {
                            if !item.is_string() {
                                *item = Value::String(budget.enum_string(item)?);
                            }
                        }
                        // 剔除空字符串元素
                        arr.retain(|item| item.as_str().map(|s| !s.is_empty()).unwrap_or(false));
                        has_valid_enum = !arr.is_empty();
                    }
                }
                if let Some(Value::Array(arr)) = map.get("enum") {
                    if arr.is_empty() {
                        map.remove("enum");
                    }
                }
                if has_valid_enum {
                    map.insert("type".to_string(), Value::String("string".to_string()));
                }

                // 10. [FIX #3682] Gemini Protobuf 终态铁律守卫：所有终态为 ARRAY 的节点必须持有 items
                // 无论前面经过了 anyOf 合并、type 联合解包还是属性清洗，杜绝任何无 items 裸数组透传导致 400 INVALID_ARGUMENT
                let final_is_array = map
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|t| t.eq_ignore_ascii_case("array"))
                    .unwrap_or(false);
                if final_is_array && !map.contains_key("items") {
                    map.insert("items".to_string(), json!({ "type": "string" }));
                }
            }
        }
        Value::Array(arr) => {
            // [FIX] 递归清理数组中的每个元素
            // 这确保了所有数组类型的值（包括但不限于 anyOf、oneOf、items、enum 等）都会被递归处理
            for item in arr.iter_mut() {
                clean_json_schema_recursive(item, is_schema_node, depth + 1, budget)?;
            }
        }
        _ => {}
    }

    Ok(is_effectively_nullable)
}

/// Merge useful structure without requiring lossless constraint intersections.
fn merge_all_of(
    map: &mut serde_json::Map<String, Value>,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    merge_all_of_at_depth(map, 0, budget)
}

fn merge_all_of_at_depth(
    map: &mut serde_json::Map<String, Value>,
    depth: usize,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    budget.charge_size(1, 16)?;
    if depth > MAX_CLEAN_RECURSION_DEPTH {
        return Err("schema normalization exceeds the supported depth budget".into());
    }
    if let Some(Value::Array(branches)) = map.remove("allOf") {
        for branch in branches {
            budget.charge(&branch)?;
            if let Value::Object(mut branch) = branch {
                merge_all_of_at_depth(&mut branch, depth + 1, budget)?;
                merge_schema_constraints(map, branch, depth + 1, budget)?;
            }
        }
    }
    Ok(())
}

fn merge_schema_constraints(
    map: &mut serde_json::Map<String, Value>,
    source: serde_json::Map<String, Value>,
    depth: usize,
    budget: &mut SchemaBudget,
) -> Result<(), String> {
    budget.charge_size(1, 16)?;
    if depth > MAX_CLEAN_RECURSION_DEPTH {
        return Err("schema normalization exceeds the supported depth budget".into());
    }
    for (key, value) in source {
        let Some(existing) = map.get_mut(&key) else {
            map.insert(key, value);
            continue;
        };
        match (key.as_str(), existing, value) {
            ("properties", Value::Object(dest), Value::Object(props)) => {
                for (name, schema) in props {
                    if let Some(Value::Object(current)) = dest.get_mut(&name) {
                        if let Value::Object(child) = schema {
                            merge_schema_constraints(current, child, depth + 1, budget)?;
                        }
                    } else {
                        dest.entry(name).or_insert(schema);
                    }
                }
            }
            ("items", Value::Object(dest), Value::Object(items)) => {
                merge_schema_constraints(dest, items, depth + 1, budget)?;
            }
            ("required", Value::Array(dest), Value::Array(required)) => {
                // Repeated allOf branches can repeatedly scan a growing set.
                // Charge that work before allocating the deduplication table.
                budget.charge_size(
                    dest.len(),
                    dest.iter().filter_map(Value::as_str).map(str::len).sum(),
                )?;
                let mut seen: std::collections::HashSet<String> = dest
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect();
                for value in required {
                    if let Some(name) = value.as_str() {
                        if seen.insert(name.to_owned()) {
                            dest.push(value);
                        }
                    }
                }
            }
            ("minimum" | "minLength" | "minItems" | "minProperties", dest, value) => {
                if dest
                    .as_f64()
                    .zip(value.as_f64())
                    .is_some_and(|(a, b)| b > a)
                {
                    *dest = value;
                }
            }
            ("maximum" | "maxLength" | "maxItems" | "maxProperties", dest, value) => {
                if dest
                    .as_f64()
                    .zip(value.as_f64())
                    .is_some_and(|(a, b)| b < a)
                {
                    *dest = value;
                }
            }
            // Keep the explicit parent/first constraint for combinations Gemini
            // cannot express (e.g. two regexes). Structural siblings still merge.
            _ => {}
        }
    }
    Ok(())
}

/// 修正工具调用参数的类型，使其符合 schema 定义
///
/// 根据 schema 中的 type 定义，自动转换参数值的类型：
/// - "123" → 123 (string → number/integer)
/// - "true" → true (string → boolean)
/// - 123 → "123" (number → string)
///
/// # Arguments
/// * `args` - 工具调用的参数对象 (会被原地修改)
/// * `schema` - 工具的参数 schema 定义 (通常是 parameters 对象)
pub fn fix_tool_call_args(args: &mut Value, schema: &Value) {
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        if let Some(args_obj) = args.as_object_mut() {
            for (key, value) in args_obj.iter_mut() {
                if let Some(prop_schema) = properties.get(key) {
                    fix_single_arg_recursive(value, prop_schema);
                }
            }
        }
    }
}

/// 递归修正单个参数的类型
fn fix_single_arg_recursive(value: &mut Value, schema: &Value) {
    // 1. 处理嵌套对象 (properties)
    if let Some(nested_props) = schema.get("properties").and_then(|p| p.as_object()) {
        if let Some(value_obj) = value.as_object_mut() {
            for (key, nested_value) in value_obj.iter_mut() {
                if let Some(nested_schema) = nested_props.get(key) {
                    fix_single_arg_recursive(nested_value, nested_schema);
                }
            }
        }
        return;
    }

    // 2. 处理数组 (items)
    let schema_type = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_lowercase();
    if schema_type == "array" {
        if let Some(items_schema) = schema.get("items") {
            if let Some(arr) = value.as_array_mut() {
                for item in arr {
                    fix_single_arg_recursive(item, items_schema);
                }
            }
        }
        return;
    }

    // 3. 处理基础类型修正
    match schema_type.as_str() {
        "number" | "integer" => {
            // 字符串 → 数字
            if let Some(s) = value.as_str() {
                // [SAFETY] 保护具有前导零的版本号或代码 (如 "01", "007")，不应转为数字
                if s.starts_with('0') && s.len() > 1 && !s.starts_with("0.") {
                    return;
                }

                // 优先尝试解析为整数
                if let Ok(i) = s.parse::<i64>() {
                    *value = Value::Number(serde_json::Number::from(i));
                } else if let Ok(f) = s.parse::<f64>() {
                    if let Some(n) = serde_json::Number::from_f64(f) {
                        *value = Value::Number(n);
                    }
                }
            }
        }
        "boolean" => {
            // 字符串 → 布尔
            if let Some(s) = value.as_str() {
                match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => *value = Value::Bool(true),
                    "false" | "0" | "no" | "off" => *value = Value::Bool(false),
                    _ => {}
                }
            } else if let Some(n) = value.as_i64() {
                // 数字 1/0 -> 布尔
                if n == 1 {
                    *value = Value::Bool(true);
                } else if n == 0 {
                    *value = Value::Bool(false);
                }
            }
        }
        "string" => {
            // 非字符串 → 字符串 (防止客户端误传数字给文本字段)
            if !value.is_string() && !value.is_null() && !value.is_object() && !value.is_array() {
                *value = Value::String(value.to_string());
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn r19_reference_keyword_as_business_property_is_preserved() {
        let mut schema = serde_json::json!({"type":"object","properties":{"$ref":{"type":"string"},"cache_control":{"type":"string"}},"required":["$ref","cache_control"]});
        super::validate_tool_schema(&schema).unwrap();
        let original = schema.clone();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema, original);
    }

    #[test]
    fn reference_limitations_use_bounded_fallbacks() {
        for (mut schema, expected_type) in [
            (json!({"$ref":"#/$defs/missing"}), "string"),
            (
                json!({"$defs":{"A":{"type":"object","properties":{"next":{"$ref":"#/$defs/A"}}}},"$ref":"#/$defs/A"}),
                "object",
            ),
            (
                json!({"$defs":{"A":{"type":"string"}},"$ref":"#/$defs/A","type":"integer"}),
                "integer",
            ),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert_eq!(schema["type"], expected_type);
            assert!(schema.get("$ref").is_none());
            if expected_type == "object" {
                assert_eq!(schema["properties"]["next"]["type"], "object");
                assert!(schema["properties"]["next"].get("$ref").is_none());
            }
        }
    }

    #[test]
    fn composed_references_keep_structure_with_a_stable_conflict_fallback() {
        let conflicting = serde_json::json!({"$defs":{"A":{"type":"object","properties":{"x":{"type":"string"}}}},"allOf":[{"$ref":"#/$defs/A"},{"properties":{"x":{"type":"integer"}}}]});
        super::validate_tool_schema(&conflicting).unwrap();
        let mut fallback = conflicting;
        super::clean_json_schema(&mut fallback).unwrap();
        assert_eq!(fallback["properties"]["x"]["type"], "string");
        let disjoint = serde_json::json!({"$defs":{"A":{"type":"string"},"B":{"type":"integer"}},"oneOf":[{"$ref":"#/$defs/A"},{"$ref":"#/$defs/B"}]});
        super::validate_tool_schema(&disjoint).unwrap();
        let mut cleaned = disjoint;
        super::clean_json_schema(&mut cleaned).unwrap();
        assert_eq!(
            cleaned,
            serde_json::json!({"anyOf":[{"type":"string"},{"type":"integer"}]})
        );
    }

    #[test]
    fn r19_parent_and_single_alternative_constraints_both_survive() {
        let mut schema = serde_json::json!({"required":["a"],"anyOf":[{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["b"]}]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema["required"], serde_json::json!(["a"]));
        assert_eq!(schema["anyOf"][0]["required"], serde_json::json!(["b"]));
        let mut schema =
            serde_json::json!({"minLength":1,"anyOf":[{"type":"string","minLength":5}]});
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema["minLength"], 1);
        assert_eq!(schema["anyOf"][0]["minLength"], 5);
    }

    #[test]
    fn type_arrays_keep_alternatives_and_allof_merges_useful_fields() {
        let mut schema = serde_json::json!({"type":["string","integer"]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            serde_json::json!({"anyOf":[{"type":"string"},{"type":"integer"}]})
        );
        for mut schema in [
            serde_json::json!({"allOf":[{"type":"string"},{"type":"integer"}]}),
            serde_json::json!({"allOf":[{"properties":{"a":{"type":"string"}}},{"properties":{"a":{"type":"integer"}}}]}),
            serde_json::json!({"additionalProperties":false,"allOf":[{"properties":{"a":{"type":"string"}}}]}),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert!(schema.get("allOf").is_none());
            assert!(schema.get("additionalProperties").is_none());
            assert!(schema["type"] == "string" || schema["properties"]["a"]["type"] == "string");
        }
        let mut schema = serde_json::json!({"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"]}]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema["properties"]["a"]["type"], "string");
        assert_eq!(schema["properties"]["b"]["type"], "integer");
        let req = schema["required"].as_array().unwrap();
        assert_eq!(req.len(), 2);
        assert!(req.contains(&serde_json::json!("a")) && req.contains(&serde_json::json!("b")));
    }

    #[test]
    fn r19_type_union_keeps_constraints_on_their_applicable_branch() {
        for (mut schema, expected) in [
            (
                json!({"type":["array","null"],"items":{"type":"integer"},"minItems":1}),
                json!({"type":"array","items":{"type":"integer"},"minItems":1,"nullable":true}),
            ),
            (
                json!({"type":["array","object"],"items":{"type":"integer"},"minItems":1,"properties":{"x":{"type":"integer"}},"required":["x"],"maxProperties":2}),
                json!({"anyOf":[{"type":"array","items":{"type":"integer"},"minItems":1},{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"maxProperties":2}]}),
            ),
            (
                json!({"type":["string","integer","boolean"],"minLength":2,"pattern":"^a","minimum":3}),
                json!({"anyOf":[{"type":"string","minLength":2,"pattern":"^a"},{"type":"integer","minimum":3},{"type":"boolean"}]}),
            ),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert_eq!(schema, expected);
        }
    }

    #[test]
    fn r19_compositions_normalize_before_structural_type_inference() {
        for (mut schema, expected) in [
            (
                json!({"allOf":[{"type":["string","integer"]}]}),
                json!({"anyOf":[{"type":"string"},{"type":"integer"}]}),
            ),
            (
                json!({"allOf":[{"type":["array","null"]},{"items":{"type":"integer"}}]}),
                json!({"type":"array","items":{"type":"integer"},"nullable":true}),
            ),
            (
                json!({"oneOf":[{"type":"string"},{"type":"integer"}],"properties":{"x":{"type":"boolean"}}}),
                json!({"anyOf":[{"type":"string"},{"type":"integer"}],"properties":{"x":{"type":"boolean"}}}),
            ),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert_eq!(schema, expected);
        }
    }

    #[test]
    fn composed_unions_keep_the_explicit_usable_alternative() {
        for mut schema in [
            json!({"allOf":[{"type":["string","integer"]}],"anyOf":[{"type":"integer"}]}),
            json!({"allOf":[{"type":["string","integer"]},{"oneOf":[{"type":"integer"}]}]}),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert_eq!(schema, json!({"type":"integer"}));
        }
    }

    #[test]
    fn r19_union_parent_items_prevent_incompatible_array_defaults() {
        for keyword in ["anyOf", "oneOf"] {
            let mut schema =
                json!({keyword:[{"type":"array"},{"type":"null"}],"items":{"type":"integer"}});
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert_eq!(
                schema,
                json!({"anyOf":[{"type":"array","items":{"type":"integer"}},{"type":"null"}],"items":{"type":"integer"}})
            );
        }
        let mut schema = json!({"allOf":[{"anyOf":[{"type":"array"},{"type":"object","properties":{"y":{"type":"boolean"}}}]},{"items":{"type":"integer"},"properties":{"x":{"type":"string"}},"required":["x"]}]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            json!({"anyOf":[{"type":"array","items":{"type":"integer"}},{"type":"object","properties":{"y":{"type":"boolean"}}}],"items":{"type":"integer"},"properties":{"x":{"type":"string"}},"required":["x"]})
        );
    }

    #[test]
    fn r19_union_parent_and_explicit_branch_items_remain_conjoined() {
        let mut schema = json!({"anyOf":[{"type":"array","items":{"type":"integer","minimum":2}},{"anyOf":[{"type":"array"},{"type":"null"}]}],"items":{"type":"integer","maximum":5}});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema["items"], json!({"type":"integer","maximum":5}));
        assert_eq!(
            schema["anyOf"][0]["items"],
            json!({"type":"integer","minimum":2})
        );
        assert_eq!(
            schema["anyOf"][1]["anyOf"][0]["items"],
            json!({"type":"integer","maximum":5})
        );
    }

    #[test]
    fn r19_nullable_union_properties_remain_required() {
        let mut schema = json!({"type":"object","properties":{"a":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["a"]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            json!({"type":"object","properties":{"a":{"type":"string","nullable":true}},"required":["a"]})
        );
    }

    #[test]
    fn oneof_keeps_alternatives_and_relaxes_exclusivity() {
        let mut schema = serde_json::json!({"oneOf":[{"type":"string"},{"type":"integer"}]});
        super::validate_tool_schema(&schema).unwrap();
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            serde_json::json!({"anyOf":[{"type":"string"},{"type":"integer"}]})
        );
        for mut schema in [
            serde_json::json!({"oneOf":[{"type":"number"},{"type":"integer"}]}),
            serde_json::json!({"oneOf":[{"type":"object","properties":{"a":{"type":"string"}}},{"type":"object","properties":{"b":{"type":"string"}}}]}),
        ] {
            super::validate_tool_schema(&schema).unwrap();
            super::clean_json_schema(&mut schema).unwrap();
            assert!(schema.get("oneOf").is_none());
            assert_eq!(schema["anyOf"].as_array().unwrap().len(), 2);
        }
    }

    #[test]
    fn r19_anyof_does_not_relax_parent_constraints_or_lose_null_only_case() {
        let mut schema =
            serde_json::json!({"type":"string","anyOf":[{"type":"number"},{"type":"null"}]});
        super::clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            serde_json::json!({"type":"string","anyOf":[{"type":"number"},{"type":"null"}]})
        );
        let mut null_only = serde_json::json!({"anyOf":[{"type":"null"}]});
        super::clean_json_schema(&mut null_only).unwrap();
        assert_eq!(null_only, serde_json::json!({"type":"null"}));
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn test_drops_boolean_subschemas() {
        // JSON Schema permits boolean sub-schemas (`prop: true|false`), but Gemini's
        // Schema proto rejects a non-object property value with HTTP 400. They must be
        // stripped at every depth (including inside `items`).
        let mut schema = json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "forbidden": false,
                        "allowed": { "type": "string" }
                    },
                    "required": ["forbidden", "allowed"]
                },
                "list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nope": false,
                            "ok": { "type": "number" }
                        }
                    }
                }
            }
        });
        clean_json_schema(&mut schema).unwrap();

        let outer_props = &schema["properties"]["outer"]["properties"];
        assert!(
            outer_props.get("forbidden").is_none(),
            "boolean sub-schema must be dropped"
        );
        assert!(
            outer_props["allowed"].is_object(),
            "valid sibling must survive"
        );

        let item_props = &schema["properties"]["list"]["items"]["properties"];
        assert!(
            item_props.get("nope").is_none(),
            "nested boolean sub-schema must be dropped"
        );
        assert!(item_props["ok"].is_object());

        let req = schema["properties"]["outer"]["required"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(req.iter().all(|r| r.as_str() != Some("forbidden")));
    }

    #[test]
    fn test_non_object_properties_normalized_to_empty_object() {
        let mut schema_null = json!({
            "type": "object",
            "properties": null,
            "required": ["foo"]
        });
        clean_json_schema(&mut schema_null).unwrap();
        assert!(schema_null["properties"].is_object());
        assert_eq!(schema_null["properties"].as_object().unwrap().len(), 0);
        assert!(schema_null.get("required").is_none());

        let mut schema_array = json!({
            "type": "object",
            "properties": ["a", "b"],
            "required": ["a"]
        });
        clean_json_schema(&mut schema_array).unwrap();
        assert!(schema_array["properties"].is_object());
        assert_eq!(schema_array["properties"].as_object().unwrap().len(), 0);
        assert!(schema_array.get("required").is_none());

        let mut schema_bool = json!({
            "type": "object",
            "properties": false
        });
        clean_json_schema(&mut schema_bool).unwrap();
        assert!(schema_bool["properties"].is_object());
        assert_eq!(schema_bool["properties"].as_object().unwrap().len(), 0);
    }
    #[test]
    fn test_clean_json_schema_draft_2020_12() {
        let mut schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                    "minLength": 1,
                    "format": "city"
                },
                // 模拟属性名冲突：pattern 是一个 Object 属性，不应被移除
                "pattern": {
                    "type": "object",
                    "properties": {
                        "regex": {
                            "type": "string",
                            "description": "Regex pattern",
                            "pattern": "^[a-z]+$"
                        }
                    }
                },
                "unit": {
                    "type": ["string", "null"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        });

        clean_json_schema(&mut schema).unwrap();

        // 1. 验证类型保持小写
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["location"]["type"], "string");

        // Supported constraints must survive the bridge unchanged.
        assert_eq!(schema["properties"]["location"]["minLength"], 1);
        assert!(schema["properties"]["location"].get("format").is_none());
        assert_eq!(
            schema["properties"]["location"]["description"],
            "The city and state, e.g. San Francisco, CA"
        );

        // 3. 验证名为 "pattern" 的属性未被误删
        assert!(schema["properties"].get("pattern").is_some());
        assert_eq!(schema["properties"]["pattern"]["type"], "object");

        assert_eq!(
            schema["properties"]["pattern"]["properties"]["regex"]["pattern"],
            "^[a-z]+$"
        );
        assert_eq!(
            schema["properties"]["pattern"]["properties"]["regex"]["description"],
            "Regex pattern"
        );

        // 5. 验证联合类型被降级为单一类型 (Protobuf 兼容性)
        assert_eq!(schema["properties"]["unit"]["type"], "string");

        // 6. 验证元数据字段被移除
        assert!(schema.get("$schema").is_none());
    }

    #[test]
    fn test_type_fallback() {
        // Test ["string", "null"] -> "string"
        let mut s1 = json!({"type": ["string", "null"]});
        clean_json_schema(&mut s1).unwrap();
        assert_eq!(s1["type"], "string");

        // Test ["integer", "null"] -> "integer" (and lowercase check if needed, though usually integer)
        let mut s2 = json!({"type": ["integer", "null"]});
        clean_json_schema(&mut s2).unwrap();
        assert_eq!(s2["type"], "integer");
    }

    #[test]
    fn test_flatten_refs() {
        let mut schema = json!({
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    }
                }
            },
            "properties": {
                "home": { "$ref": "#/$defs/Address" }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证引用被展开且类型转为小写
        assert_eq!(schema["properties"]["home"]["type"], "object");
        assert_eq!(
            schema["properties"]["home"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn test_clean_json_schema_missing_required() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "existing_prop": { "type": "string" }
            },
            "required": ["existing_prop", "missing_prop"]
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 missing_prop 被从 required 中移除
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "existing_prop");
    }

    // [NEW TEST] 验证 anyOf 类型提取
    #[test]
    fn test_anyof_type_extraction() {
        // 测试 FastMCP 风格的 Optional[str] schema
        let mut schema = json!({
            "type": "object",
            "properties": {
                "testo": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": null,
                    "title": "Testo"
                },
                "importo": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"}
                    ],
                    "default": null,
                    "title": "Importo"
                },
                "attivo": {
                    "type": "boolean",
                    "title": "Attivo"
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 anyOf 被移除
        assert!(schema["properties"]["testo"].get("anyOf").is_none());
        assert!(schema["properties"]["importo"].get("anyOf").is_none());

        // 验证 type 被正确提取
        assert_eq!(schema["properties"]["testo"]["type"], "string");
        assert_eq!(schema["properties"]["importo"]["type"], "number");
        assert_eq!(schema["properties"]["attivo"]["type"], "boolean");

        // 验证 default 被移除 (白名单之外)
        assert!(schema["properties"]["testo"].get("default").is_none());
    }

    // [NEW TEST] 验证 oneOf 类型提取
    #[test]
    fn test_oneof_type_extraction() {
        let mut schema = json!({
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "null"}
                    ]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        assert!(schema["properties"]["value"].get("oneOf").is_none());
        assert_eq!(schema["properties"]["value"]["type"], "integer");
    }

    // [NEW TEST] 验证已有 type 不被覆盖
    #[test]
    fn test_existing_type_preserved() {
        let mut schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "anyOf": [
                        {"type": "number"}
                    ]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // type 已存在，不应被 anyOf 中的类型覆盖
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(
            schema["properties"]["name"]["anyOf"],
            json!([{"type":"number"}])
        );
    }

    // [NEW TEST] 验证 Issue #815: anyOf 内部属性不丢失
    #[test]
    fn test_issue_815_anyof_properties_preserved() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "config": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "recursive": { "type": "boolean" }
                            },
                            "required": ["path"]
                        },
                        { "type": "null" }
                    ]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        let config = &schema["properties"]["config"];

        // 1. 验证类型被提取
        assert_eq!(config["type"], "object");

        // 2. 验证 anyOf 内部的 properties 被合并上来了
        assert!(config.get("properties").is_some());
        assert_eq!(config["properties"]["path"]["type"], "string");
        assert_eq!(config["properties"]["recursive"]["type"], "boolean");

        // 3. 验证 required 被合并上来了
        let req = config["required"].as_array().unwrap();
        assert!(req.iter().any(|v| v == "path"));

        // 4. 验证 anyOf 字段本身被移除
        assert!(config.get("anyOf").is_none());

        // 5. 验证没有因为“空”而注入 reason (因为我们保留了属性)
        assert!(config["properties"].get("reason").is_none());
    }

    // [NEW TEST] 验证安全检查：不应处理非 Schema 对象（保护工具调用）
    #[test]
    fn test_clean_json_schema_on_non_schema_object() {
        // 模拟包含 description、command、type 等常规业务字段的运行时工具调用对象
        let mut tool_call = json!({
            "functionCall": {
                "name": "run_command",
                "args": {
                    "description": "Run: git status",
                    "command": "git status",
                    "shell": "default"
                },
                "id": "call_123"
            }
        });

        // 调用清洗逻辑
        clean_json_schema(&mut tool_call).unwrap();

        // 验证：非 Schema 字段与运行时实参绝对不应被剥离，command 与 description 必须同时完好保留
        let fc = &tool_call["functionCall"];
        assert_eq!(fc["name"], "run_command");
        assert_eq!(fc["args"]["command"], "git status");
        assert_eq!(fc["args"]["description"], "Run: git status");
        assert_eq!(fc["args"]["shell"], "default");
        assert_eq!(fc["id"], "call_123");
    }

    // [NEW TEST] 验证 Nullable 处理
    #[test]
    fn test_nullable_handling_with_description() {
        let mut schema = json!({
            "type": ["string", "null"],
            "description": "User name"
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 type 被降级，且描述被追加 (nullable)
        assert_eq!(schema["type"], "string");
        assert!(schema["description"]
            .as_str()
            .unwrap()
            .contains("User name"));
        assert_eq!(schema["nullable"], true);
    }

    // [NEW TEST] 验证 anyOf 内部的 propertyNames 被移除
    #[test]
    fn test_clean_anyof_with_propertynames() {
        let mut schema = json!({
            "properties": {
                "config": {
                    "anyOf": [
                        {
                            "type": "object",
                            "propertyNames": {"pattern": "^[a-z]+$"},
                            "properties": {
                                "key": {"type": "string"}
                            }
                        },
                        {"type": "null"}
                    ]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 anyOf 被移除（已被合并）
        let config = &schema["properties"]["config"];
        assert!(config.get("anyOf").is_none());

        // 验证 propertyNames 被移除
        assert!(config.get("propertyNames").is_none());

        // 验证合并后的 properties 存在且没有 propertyNames
        assert!(config.get("properties").is_some());
        assert_eq!(config["properties"]["key"]["type"], "string");
    }

    // [NEW TEST] 验证 items 数组中的 const 被移除
    #[test]
    fn test_clean_items_array_with_const() {
        let mut schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "status": {
                        "const": "active",
                        "type": "string"
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 const 被移除
        let status = &schema["items"]["properties"]["status"];
        assert!(status.get("const").is_none());

        // 验证 type 仍然存在
        assert_eq!(status["type"], "string");
    }

    // [NEW TEST] 验证多层嵌套数组的清理
    #[test]
    fn test_deep_nested_array_cleaning() {
        let mut schema = json!({
            "properties": {
                "data": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "propertyNames": {"maxLength": 10},
                                        "const": "test",
                                        "properties": {
                                            "name": {"type": "string"}
                                        }
                                    },
                                    {"type": "null"}
                                ]
                            }
                        }
                    ]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证深层嵌套的非法字段都被移除
        let data = &schema["properties"]["data"];

        // anyOf 应该被合并移除
        assert!(data.get("anyOf").is_none());

        // 验证没有 propertyNames 和 const 逃逸到顶层
        assert!(data.get("propertyNames").is_none());
        assert!(data.get("const").is_none());

        // 验证结构被正确保留
        assert_eq!(data["type"], "array");
        if let Some(items) = data.get("items") {
            // items 内部的 anyOf 也应该被合并
            assert!(items.get("anyOf").is_none());
            assert!(items.get("propertyNames").is_none());
            assert!(items.get("const").is_none());
        }
    }

    #[test]
    fn test_fix_tool_call_args() {
        let mut args = serde_json::json!({
            "port": "8080",
            "enabled": "true",
            "timeout": "5.5",
            "metadata": {
                "retry": "3"
            },
            "tags": ["1", "2"]
        });

        let schema = serde_json::json!({
            "properties": {
                "port": { "type": "integer" },
                "enabled": { "type": "boolean" },
                "timeout": { "type": "number" },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "retry": { "type": "integer" }
                    }
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "integer" }
                }
            }
        });

        fix_tool_call_args(&mut args, &schema);

        assert_eq!(args["port"], 8080);
        assert_eq!(args["enabled"], true);
        assert_eq!(args["timeout"], 5.5);
        assert_eq!(args["metadata"]["retry"], 3);
        assert_eq!(args["tags"], serde_json::json!([1, 2]));
    }

    #[test]
    fn test_fix_tool_call_args_protection() {
        let mut args = serde_json::json!({
            "version": "01.0",
            "code": "007"
        });

        let schema = serde_json::json!({
            "properties": {
                "version": { "type": "number" },
                "code": { "type": "integer" }
            }
        });

        fix_tool_call_args(&mut args, &schema);

        // 应保留字符串以防破坏语义
        assert_eq!(args["version"], "01.0");
        assert_eq!(args["code"], "007");
    }

    // [NEW TEST #952] 验证嵌套层级的 $defs 能被正确收集和展开
    #[test]
    fn test_nested_defs_flattening() {
        // MCP 工具常常将 $defs 嵌套在 properties 内部，而非根层级
        let mut schema = json!({
            "type": "object",
            "properties": {
                "config": {
                    "$defs": {
                        "Address": {
                            "type": "object",
                            "properties": {
                                "city": { "type": "string" },
                                "zip": { "type": "string" }
                            }
                        }
                    },
                    "type": "object",
                    "properties": {
                        "home": { "$ref": "#/$defs/Address" },
                        "work": { "$ref": "#/$defs/Address" }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证嵌套的 $ref 被正确解析
        let home = &schema["properties"]["config"]["properties"]["home"];
        assert_eq!(
            home["type"], "object",
            "home should have type 'object' from resolved $ref"
        );
        assert_eq!(
            home["properties"]["city"]["type"], "string",
            "home.properties.city should exist from resolved Address"
        );

        // 验证没有残留的 $ref
        assert!(
            home.get("$ref").is_none(),
            "home should not have orphan $ref"
        );

        // 验证 work 也被正确解析
        let work = &schema["properties"]["config"]["properties"]["work"];
        assert_eq!(work["type"], "object");
        assert!(work.get("$ref").is_none());
    }

    // [NEW TEST #952] 验证无法解析的 $ref 被优雅降级
    #[test]
    fn test_unresolved_ref_fallback() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "external": { "$ref": "https://example.com/schemas/External.json" },
                "missing": { "$ref": "#/$defs/NonExistent" }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证外部引用被降级为 string 类型
        let external = &schema["properties"]["external"];
        assert_eq!(
            external["type"], "string",
            "unresolved external $ref should fallback to string"
        );
        assert!(
            external["description"]
                .as_str()
                .unwrap()
                .contains("Unresolved $ref"),
            "description should contain unresolved $ref hint"
        );

        // 验证内部缺失引用也被降级
        let missing = &schema["properties"]["missing"];
        assert_eq!(missing["type"], "string");
        assert!(missing["description"]
            .as_str()
            .unwrap()
            .contains("NonExistent"));
    }

    // [NEW TEST #952] 验证深层嵌套的多级 $defs 都能被收集
    #[test]
    fn test_deeply_nested_multi_level_defs() {
        let mut schema = json!({
            "type": "object",
            "$defs": {
                "RootDef": { "type": "integer" }
            },
            "properties": {
                "level1": {
                    "type": "object",
                    "$defs": {
                        "Level1Def": { "type": "boolean" }
                    },
                    "properties": {
                        "level2": {
                            "type": "object",
                            "$defs": {
                                "Level2Def": { "type": "number" }
                            },
                            "properties": {
                                "useRoot": { "$ref": "#/$defs/RootDef" },
                                "useLevel1": { "$ref": "#/$defs/Level1Def" },
                                "useLevel2": { "$ref": "#/$defs/Level2Def" }
                            }
                        }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        let level2_props = &schema["properties"]["level1"]["properties"]["level2"]["properties"];

        // 验证所有层级的 $defs 都被正确解析
        assert_eq!(
            level2_props["useRoot"]["type"], "integer",
            "RootDef should resolve"
        );
        assert_eq!(
            level2_props["useLevel1"]["type"], "boolean",
            "Level1Def should resolve"
        );
        assert_eq!(
            level2_props["useLevel2"]["type"], "number",
            "Level2Def should resolve"
        );

        // 验证没有残留 $ref
        assert!(level2_props["useRoot"].get("$ref").is_none());
        assert!(level2_props["useLevel1"].get("$ref").is_none());
        assert!(level2_props["useLevel2"].get("$ref").is_none());
    }

    // [NEW TEST] 验证对非标准字段（如 cornerRadius）的清洗和启发式修复
    #[test]
    fn test_non_standard_field_cleaning_and_healing() {
        let mut schema = json!({
            "type": "array",
            "items": {
                "cornerRadius": { "type": "number" },
                "fillColor": { "type": "string" }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 items 中的非标准字段被移动到了 properties 内部，并增加了 type: object
        let items = &schema["items"];
        assert_eq!(
            items["type"], "object",
            "Malformed items should be healed to type object"
        );
        assert!(
            items.get("properties").is_some(),
            "Malformed items should have properties object"
        );
        assert_eq!(items["properties"]["cornerRadius"]["type"], "number");
        assert_eq!(items["properties"]["fillColor"]["type"], "string");

        // 验证原始字段已从 items 顶层移除（白名单过滤）
        assert!(items.get("cornerRadius").is_none());
        assert!(items.get("fillColor").is_none());
    }

    // [NEW TEST] 验证隐式 Array (只有 items) 和隐式 Object (只有 properties) 的处理
    #[test]
    fn test_implicit_type_injection() {
        let mut schema = json!({
            "properties": {
                "values": {
                    "items": {
                        "cornerRadius": { "type": "number" }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 values 被注入了 type: array
        assert_eq!(schema["properties"]["values"]["type"], "array");

        // 验证 items 被启发式修复为 type: object 并包含 properties
        let items = &schema["properties"]["values"]["items"];
        assert_eq!(items["type"], "object");
        assert!(items["properties"].get("cornerRadius").is_some());
    }

    #[test]
    fn test_gemini_strict_validation_injection() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "patterns": {
                    "items": {
                        "properties": {
                            "type": {
                                "enum": ["A", "B"]
                            }
                        }
                    }
                },
                "nested_props": {
                    "properties": {
                        "foo": { "type": "string" }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 enum 自动补全了 type: string
        let type_node = &schema["properties"]["patterns"]["items"]["properties"]["type"];
        assert_eq!(type_node["type"], "string");
        assert!(type_node.get("enum").is_some());

        // 验证 嵌套 properties 自动补全了 type: object
        assert_eq!(schema["properties"]["nested_props"]["type"], "object");

        // 验证 patterns 自动补全了 type: array
        assert_eq!(schema["properties"]["patterns"]["type"], "array");
    }
    #[test]
    fn test_malformed_items_as_properties() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "items": {
                        "color": { "type": "string" },
                        "size": { "type": "number" }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 验证 items 被移除并转换为 properties
        let config = &schema["properties"]["config"];
        assert!(config.get("items").is_none());
        assert_eq!(config["properties"]["color"]["type"], "string");
        assert_eq!(config["properties"]["size"]["type"], "number");
        assert_eq!(config["type"], "object");
    }

    #[test]
    fn test_circular_ref_flattening() {
        // 模拟循环引用：A -> B, B -> A
        let mut schema = json!({
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {
                        "toB": { "$ref": "#/$defs/B" }
                    }
                },
                "B": {
                    "type": "object",
                    "properties": {
                        "toA": { "$ref": "#/$defs/A" }
                    }
                }
            },
            "properties": {
                "start": { "$ref": "#/$defs/A" }
            }
        });

        // 如果没有深度限制，这里会发生栈溢出
        // 有了深度限制，它应该能正常返回（尽管展开是不完整的）
        clean_json_schema(&mut schema).unwrap();

        // 验证基本结构保留，没有崩溃
        assert_eq!(schema["properties"]["start"]["type"], "object");
        assert!(schema["properties"]["start"]["properties"]
            .get("toB")
            .is_some());
    }

    #[test]
    fn test_any_of_preserves_all_non_null_branches() {
        let mut schema = json!({
            "anyOf": [
                { "type": "string" },
                { "type": "object", "properties": { "foo": { "type": "string" } } },
                { "type": "null" }
            ]
        });

        clean_json_schema(&mut schema).unwrap();

        assert_eq!(schema["anyOf"].as_array().unwrap().len(), 2);
        assert_eq!(schema["anyOf"][0]["type"], "string");
        assert_eq!(schema["anyOf"][1]["type"], "object");
        assert_eq!(schema["anyOf"][1]["properties"]["foo"]["type"], "string");
        assert_eq!(schema["nullable"], true);
    }

    #[test]
    fn test_issue_3327_const_normalization() {
        // 场景 1: 基础 const 转换
        let mut schema1 = json!({
            "type": "object",
            "properties": {
                "action_type": {
                    "const": "element"
                },
                "count": {
                    "const": 5
                },
                "enabled": {
                    "const": true
                }
            }
        });

        clean_json_schema(&mut schema1).unwrap();

        assert_eq!(schema1["properties"]["action_type"]["type"], "string");
        assert_eq!(
            schema1["properties"]["action_type"]["enum"],
            json!(["element"])
        );
        assert!(schema1["properties"]["action_type"].get("const").is_none());

        assert_eq!(schema1["properties"]["count"]["type"], "string");
        assert_eq!(schema1["properties"]["count"]["enum"], json!(["5"]));
        assert!(schema1["properties"]["count"].get("const").is_none());

        assert_eq!(schema1["properties"]["enabled"]["type"], "string");
        assert_eq!(schema1["properties"]["enabled"]["enum"], json!(["true"]));
        assert!(schema1["properties"]["enabled"].get("const").is_none());

        // 场景 2: ZCode Computer Use MCP 的 anyOf 联合嵌套包含 const
        let mut schema2 = json!({
            "type": "object",
            "properties": {
                "target": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "const": "element"
                                },
                                "state_id": {
                                    "type": "string"
                                },
                                "index": {
                                    "type": "integer"
                                }
                            },
                            "required": ["type", "state_id", "index"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "const": "coordinate"
                                },
                                "x": {
                                    "type": "integer"
                                },
                                "y": {
                                    "type": "integer"
                                }
                            },
                            "required": ["type", "x", "y"],
                            "additionalProperties": false
                        }
                    ]
                }
            }
        });

        clean_json_schema(&mut schema2).unwrap();

        let branches = schema2["properties"]["target"]["anyOf"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
        for (branch, discriminator) in branches.iter().zip(["element", "coordinate"]) {
            let target_props = &branch["properties"];
            assert_eq!(target_props["type"]["type"], "string");
            assert_eq!(target_props["type"]["enum"], json!([discriminator]));
            assert!(target_props["type"].get("const").is_none());
            assert!(target_props["type"].get("properties").is_none());
        }
    }

    #[test]
    fn test_nested_array_without_items_gets_gemini_fallback() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "object",
                    "properties": {
                        "where": {
                            "type": "array",
                            "items": { "type": "array" }
                        }
                    }
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        assert_eq!(
            schema["properties"]["query"]["properties"]["where"]["items"]["items"],
            json!({ "type": "string" })
        );
    }

    #[test]
    fn test_sanitize_description() {
        let multi_line = "This is a tool description\nwith multiple lines\r\nand   extra   spaces.";
        assert_eq!(sanitize_description(multi_line), multi_line);

        let overlong = "a".repeat(9000);
        let sanitized = sanitize_description(&overlong);
        assert!(sanitized.chars().count() <= MAX_DESCRIPTION_LENGTH);
        assert!(sanitized.ends_with("... [truncated]"));
    }

    #[test]
    fn test_clean_json_schema_ensures_properties_on_object() {
        let mut schema = json!({
            "type": "OBJECT",
            "description": "Some description\nwith newlines"
        });

        clean_json_schema(&mut schema).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"], json!({}));
        assert_eq!(schema["description"], "Some description\nwith newlines");
    }

    #[test]
    fn test_issue_2041_gemini_enum_string_and_non_empty_constraints() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "safesearch": {
                    "type": "string",
                    "enum": ["", "moderate", "strict"]
                },
                "level": {
                    "type": "integer",
                    "enum": [1, 2, 3]
                },
                "active": {
                    "type": "boolean",
                    "enum": [true]
                },
                "empty_only": {
                    "type": "string",
                    "enum": [""]
                }
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 1. 空字符串 enum 必须被剔除 (Gemini: Schema.enum[i]: cannot be empty)
        assert_eq!(
            schema["properties"]["safesearch"]["enum"],
            json!(["moderate", "strict"])
        );

        // 2. 剔除空字符串后若 enum 为空，则必须移除 enum 字段
        assert!(schema["properties"]["empty_only"].get("enum").is_none());

        // 3. 非 string 类型如果包含 enum，类型必须归一化为 string (Gemini: Schema.enum: only allowed for STRING type)
        assert_eq!(schema["properties"]["level"]["type"], "string");
        assert_eq!(
            schema["properties"]["level"]["enum"],
            json!(["1", "2", "3"])
        );

        assert_eq!(schema["properties"]["active"]["type"], "string");
        assert_eq!(schema["properties"]["active"]["enum"], json!(["true"]));
    }

    #[test]
    fn test_issue_3682_deeply_nested_schema_bare_array_fallback() {
        // [FIX #3682] 验证深度嵌套树结构中，位于深度 > 10 的裸数组 (无 items) 能被正确兜底注入 items
        fn build_mindmap_node(level: usize) -> Value {
            if level == 1 {
                json!({
                    "type": "object",
                    "properties": {
                        "children": { "type": "array" } // 裸数组，无 items
                    }
                })
            } else {
                json!({
                    "type": "object",
                    "properties": {
                        "children": {
                            "type": "array",
                            "items": build_mindmap_node(level - 1)
                        }
                    }
                })
            }
        }

        let mut schema = json!({
            "type": "object",
            "properties": {
                "root": build_mindmap_node(7)
            }
        });

        clean_json_schema(&mut schema).unwrap();

        // 沿树遍历到最深层的 children 节点
        let mut curr = &schema["properties"]["root"];
        for _ in 1..7 {
            assert_eq!(curr["type"], "object");
            assert_eq!(curr["properties"]["children"]["type"], "array");
            curr = &curr["properties"]["children"]["items"];
        }

        // 最底层的 children 必须被补齐 items: { type: "string" }
        assert_eq!(curr["type"], "object");
        assert_eq!(curr["properties"]["children"]["type"], "array");
        assert_eq!(
            curr["properties"]["children"]["items"],
            json!({ "type": "string" })
        );
    }

    #[test]
    fn test_issue_3682_failsafe_fallback_beyond_max_recursion_depth() {
        // 构造超过 MAX_CLEAN_RECURSION_DEPTH (64) 的极端深层结构 (70 层)
        fn build_deep_chain(depth: usize) -> Value {
            if depth == 1 {
                json!({
                    "type": "array" // 裸数组
                })
            } else {
                json!({
                    "type": "object",
                    "properties": {
                        "next": build_deep_chain(depth - 1)
                    }
                })
            }
        }

        let mut schema = build_deep_chain(70);
        clean_json_schema(&mut schema).unwrap();

        // 验证即便超过 64 层递归，迭代式的 apply_minimal_failsafe 也保障了最深层的裸 array 拥有 items 且无栈溢出
        let mut curr = &schema;
        for _ in 1..70 {
            curr = &curr["properties"]["next"];
        }
        assert_eq!(curr["type"], "array");
        assert_eq!(curr["items"], json!({ "type": "string" }));
    }

    #[test]
    fn test_issue_3690_cyclic_binary_tree_no_oom() {
        // [FIX #3690] 模拟带有分支因子 b=2 的递归二叉树 Schema。
        // 在修复前，未记录 active_refs 的 64 层递归会导致 2^64 指数级节点复制展开，数秒内耗尽 VPS 内存触发 OOM (SIGKILL 137)。
        // 修复后，活跃祖先检测必须在发现循环引用后立即安全降级并终结递归，且耗时小于 5ms。
        let mut schema = json!({
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" },
                        "left": { "$ref": "#/$defs/TreeNode" },
                        "right": { "$ref": "#/$defs/TreeNode" }
                    }
                }
            },
            "type": "object",
            "properties": {
                "root": { "$ref": "#/$defs/TreeNode" }
            }
        });

        let start = std::time::Instant::now();
        clean_json_schema(&mut schema).unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 50,
            "Recursive schema flattening took too long: {:?}",
            elapsed
        );

        let root = &schema["properties"]["root"];
        assert_eq!(root["type"], "object");
        assert_eq!(root["properties"]["value"]["type"], "string");

        // 验证子节点的循环引用已被平滑降级，且标记了 Cyclic $ref 提示
        let left = &root["properties"]["left"];
        assert_eq!(left["type"], "object");
        assert!(
            left["description"]
                .as_str()
                .unwrap_or("")
                .contains("Cyclic $ref"),
            "Left child should contain Cyclic $ref hint in description"
        );

        let right = &root["properties"]["right"];
        assert_eq!(right["type"], "object");
        assert!(
            right["description"]
                .as_str()
                .unwrap_or("")
                .contains("Cyclic $ref"),
            "Right child should contain Cyclic $ref hint in description"
        );
    }

    #[test]
    fn test_issue_3682_array_with_properties_preserves_items_and_strips_properties() {
        // [FIX #3682] 当节点同时携带 type: array 和 properties 时，
        // Gemini Protobuf 约束：ARRAY 禁止包含 properties，且必须包含 items。
        // 必须剥离 properties 并保留/补全 items，严禁误将 items 移除。
        let mut schema = json!({
            "type": "array",
            "properties": {
                "ignored_key": { "type": "string" }
            },
            "items": { "type": "number" }
        });

        clean_json_schema(&mut schema).unwrap();

        assert_eq!(schema["type"], "array");
        assert!(
            schema.get("properties").is_none(),
            "properties must be stripped from array schema"
        );
        assert_eq!(schema["items"]["type"], "number");
    }

    #[test]
    fn test_union_array_type_gets_items_fallback() {
        // 验证带有联合类型的数组 (type: ["array", "null"]) 即使未显式声明 items，
        // 也能在解包为 type: "array" 后正确兜底注入 items: { type: "string" }
        let mut schema = json!({
            "type": ["array", "null"]
        });

        clean_json_schema(&mut schema).unwrap();

        assert_eq!(schema["type"], "array");
        assert_eq!(schema["items"], json!({ "type": "string" }));
        assert_eq!(schema["nullable"], true);
    }
}

#[cfg(test)]
mod resource_budget_tests {
    use super::*;

    fn shared_tree(depth: usize, leaf: Value) -> Value {
        let mut defs = serde_json::Map::new();
        defs.insert("D0".into(), leaf);
        for i in 1..=depth {
            let reference = format!("#/$defs/D{}", i - 1);
            defs.insert(
                format!("D{i}"),
                json!({"type":"object", "properties": {
                    "left": {"$ref":reference}, "right": {"$ref":reference}
                }}),
            );
        }
        json!({"$defs": defs, "$ref": format!("#/$defs/D{depth}")})
    }

    #[test]
    fn shared_acyclic_references_are_bounded_before_materialization() {
        let schema = shared_tree(20, json!({"type":"string"}));
        assert!(schema.to_string().len() < 4096);
        assert!(validate_tool_schema(&schema)
            .unwrap_err()
            .contains("budget"));
        let mut cleaned = schema.clone();
        assert!(clean_json_schema(&mut cleaned)
            .unwrap_err()
            .contains("budget"));
        assert_eq!(
            cleaned, schema,
            "rejected expansion must not mutate the input"
        );
    }

    #[test]
    fn shared_payload_bytes_are_bounded_for_response_schemas_too() {
        for leaf in [
            json!({"type":"string", "description":"x".repeat(8192)}),
            json!({"type":"string", "enum":["y".repeat(8192)]}),
        ] {
            let schema = shared_tree(8, leaf);
            assert!(validate_tool_schema(&schema)
                .unwrap_err()
                .contains("budget"));
            let mut cleaned = schema.clone();
            assert!(clean_response_schema(&mut cleaned)
                .unwrap_err()
                .contains("budget"));
            assert_eq!(cleaned, schema);
        }
    }

    #[test]
    fn ordinary_shared_definitions_remain_supported() {
        let mut schema = shared_tree(3, json!({"type":"string", "enum":["a","b"]}));
        validate_tool_schema(&schema).unwrap();
        clean_json_schema(&mut schema).unwrap();
        assert!(!schema.to_string().contains("$ref"));
        assert_eq!(
            schema["properties"]["left"]["properties"]["right"]["properties"]["left"]["enum"],
            json!(["a", "b"])
        );
    }

    fn nested_array_union(depth: usize, mut leaf: Value) -> Value {
        for _ in 0..depth {
            leaf = json!({
                "type":"array", "items":leaf,
                "anyOf":[{"type":"array","minItems":1},{"type":"array","maxItems":2}]
            });
        }
        leaf
    }

    fn assert_normalization_budget_rejected(schema: Value) {
        assert!(validate_tool_schema(&schema)
            .unwrap_err()
            .contains("budget"));
        for clean in [clean_json_schema, clean_response_schema] {
            let mut output = schema.clone();
            assert!(clean(&mut output).unwrap_err().contains("budget"));
            assert_eq!(
                output, schema,
                "budget failure must preserve the original schema"
            );
        }
    }

    #[test]
    fn nested_anyof_items_cannot_bypass_budget_without_references() {
        let schema = nested_array_union(9, json!({"type":"string"}));
        assert_eq!(schema.to_string().len(), 872);
        assert_normalization_budget_rejected(schema);
    }

    #[test]
    fn repeated_type_constraints_cannot_amplify_without_references() {
        let mut schema = json!({"type":"string"});
        for _ in 0..16 {
            schema = json!({"type":["object","object"],"properties":{"child":schema}});
        }
        assert_normalization_budget_rejected(schema);
    }

    #[test]
    fn inherited_description_and_enum_payloads_share_the_byte_budget() {
        for leaf in [
            json!({"type":"string", "description":"x".repeat(160_000)}),
            json!({"type":"string", "enum":["y".repeat(160_000)]}),
        ] {
            assert_normalization_budget_rejected(nested_array_union(2, leaf));
        }
    }

    #[test]
    fn enum_serialization_is_bounded_before_escaping_expands_the_buffer() {
        let schema = json!({"type":"string", "enum":[{"payload":"\u{0001}".repeat(200_000)}]});
        // Raw strings fit the input budget; JSON escaping the enum object does not.
        SchemaBudget::default().charge(&schema).unwrap();
        assert_normalization_budget_rejected(schema);
    }

    #[test]
    fn small_nested_unions_preserve_items_and_branch_constraints() {
        let original = nested_array_union(2, json!({"type":"string", "enum":["a","b"]}));
        validate_tool_schema(&original).unwrap();
        let mut cleaned = original;
        clean_json_schema(&mut cleaned).unwrap();
        assert_eq!(cleaned["anyOf"][0]["minItems"], 1);
        assert_eq!(cleaned["anyOf"][1]["maxItems"], 2);
        for branch in cleaned["anyOf"].as_array().unwrap() {
            assert_eq!(branch["items"], cleaned["items"]);
        }
        assert_eq!(
            cleaned["items"]["anyOf"][0]["items"]["enum"],
            json!(["a", "b"])
        );
        SchemaBudget::default().charge(&cleaned).unwrap();
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;

    #[test]
    fn send_message_patterns_remain_usable_at_nested_tool_paths() {
        let constraint = json!({"allOf":[
            {"pattern":"^[^\\n\\r]*$"}, {"pattern":"^[\\s\\S]{0,300}$"}
        ]});
        let mut schema = json!({"type":"object","properties":{
            "to":constraint,
            "message":{"anyOf":[{"type":"string"},{"anyOf":[
                {"type":"null"},
                {"type":"object","properties":{"request_id":constraint}},
                {"type":"object","properties":{"request_id":constraint}}
            ]}]}
        }});
        validate_tool_schema(&schema).unwrap();
        clean_json_schema_for_tool(&mut schema, "SendMessage").unwrap();
        let expected = json!({"type":"string","pattern":"^[^\\n\\r]*$"});
        assert_eq!(schema["properties"]["to"], expected);
        let variants = &schema["properties"]["message"]["anyOf"][1]["anyOf"];
        assert_eq!(variants[0]["properties"]["request_id"], expected);
        assert_eq!(variants[1]["properties"]["request_id"], expected);
    }

    #[test]
    fn allof_merges_nested_properties_required_and_compatible_bounds() {
        let mut schema = json!({"type":"object","additionalProperties":true,"allOf":[
            {"properties":{"name":{"type":"string","minLength":1}},"required":["name"]},
            {"allOf":[{"properties":{"name":{"minLength":3,"maxLength":30},"age":{"type":"integer"}},"required":["age"]}]},
            {"properties":{"name":{"maxLength":20}},"additionalProperties":false}
        ]});
        validate_tool_schema(&schema).unwrap();
        clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            json!({"type":"object","properties":{
            "name":{"type":"string","minLength":3,"maxLength":20},
            "age":{"type":"integer"}
        },"required":["name","age"]})
        );
    }

    #[test]
    fn tagged_object_alternatives_are_cleaned_for_tools_and_response_schemas() {
        let original = json!({"type":"object","properties":{"action":{"oneOf":[
            {"type":"object","properties":{"kind":{"const":"read"},"path":{"type":"string"}},"required":["kind","path"]},
            {"type":"object","properties":{"kind":{"const":"write"},"text":{"type":"string"}},"required":["kind","text"]}
        ]}}});
        validate_tool_schema(&original).unwrap();
        for clean in [clean_json_schema, clean_response_schema] {
            let mut schema = original.clone();
            clean(&mut schema).unwrap();
            let action = &schema["properties"]["action"];
            assert!(action.get("oneOf").is_none());
            assert_eq!(
                action["anyOf"][0]["properties"]["kind"],
                json!({"type":"string","enum":["read"]})
            );
            assert_eq!(
                action["anyOf"][1]["properties"]["kind"],
                json!({"type":"string","enum":["write"]})
            );
            assert_eq!(action["anyOf"][1]["required"], json!(["kind", "text"]));
        }
    }

    #[test]
    fn nullable_type_array_does_not_override_explicit_alternatives() {
        let mut schema = json!({"type":["string","null"],"anyOf":[
            {"type":"string","minLength":1},{"type":"null"}
        ]});
        validate_tool_schema(&schema).unwrap();
        clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema,
            json!({"type":"string","minLength":1,"nullable":true})
        );
    }

    #[test]
    fn references_use_exact_pointers_and_merge_useful_siblings() {
        let mut schema = json!({"type":"object","$defs":{"A":{"type":"integer"}},"properties":{
            "a/b~c":{"type":"string","minLength":1},
            "alias":{"$ref":"#/properties/a~1b~0c","minLength":3},
            "root":{"$ref":"#/$defs/A"},
            "nested":{"$defs":{"A":{"type":"boolean"}},"type":"object","properties":{"value":{"$ref":"#/properties/nested/$defs/A"}}}
        }});
        validate_tool_schema(&schema).unwrap();
        clean_json_schema(&mut schema).unwrap();
        assert_eq!(
            schema["properties"]["alias"],
            json!({"type":"string","minLength":3})
        );
        assert_eq!(schema["properties"]["root"]["type"], "integer");
        assert_eq!(
            schema["properties"]["nested"]["properties"]["value"]["type"],
            "boolean"
        );
        assert!(schema.get("$defs").is_none());
    }

    #[test]
    fn recursive_tree_retains_fields_with_a_finite_terminal_object() {
        let mut schema = json!({"$defs":{"Node":{"type":"object","properties":{
            "name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#/$defs/Node"}}
        },"required":["name"]}},"type":"object","properties":{"root":{"$ref":"#/$defs/Node"}}});
        validate_tool_schema(&schema).unwrap();
        clean_json_schema(&mut schema).unwrap();
        let root = &schema["properties"]["root"];
        assert_eq!(root["properties"]["name"]["type"], "string");
        assert_eq!(root["required"], json!(["name"]));
        let terminal = &root["properties"]["children"]["items"];
        assert_eq!(terminal["type"], "object");
        assert!(terminal.get("$ref").is_none());
        assert!(schema.to_string().len() < 1024);
    }
}
