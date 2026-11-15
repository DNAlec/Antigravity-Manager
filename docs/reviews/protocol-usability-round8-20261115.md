# 协议可用性第八轮修复与复审

基线 HEAD 为 `3535d6f99d75fe4faabbc1eb79370e47c53277ab`，沿用开发分支 `codex/protocol-usability-round3`。本轮审查与修复以包含[第七轮未提交修复](protocol-usability-round7-20261115.md)的当前工作树为基础，不是仅检查 HEAD。审查快照记录了 585 个受 Git 管理及未忽略的新文件的 SHA-256，修复前确认生产工作树与该快照一致。

本报告记录跨仓审查中第 3、6、7、8、9、10 项对应的 **6 项 AM 发现**，按请求和响应行为说明。第七轮改动保留；本轮没有切分支、提交、推送、版本变更、发布或部署。

## 六项修复

### 1. 客户端自定义搜索工具保留身份

Claude 请求单独声明名为 `web_search`、`google_search` 或 `builtin_web_search` 的客户端函数时，原先会仅因名称被替换成内置 `googleSearch`，丢掉自定义参数 schema，强制调用选择器却仍指向原函数。

现在工具分类优先尊重明确的 `type:custom` 或 `input_schema`，声明与选择器保持一致。真正的服务端搜索仍转换为内置搜索；旧版无 schema 搜索别名保留既有兼容行为。混合客户端工具继续按原顺序进入共享管线。

实现：[Claude 工具类型](../../src-tauri/src/proxy/mappers/claude/models.rs)、[Claude 请求转换](../../src-tauri/src/proxy/mappers/claude/request.rs)。

### 2. Markdown 源码示例保留为正文

原先媒体提取正则会将代码围栏、行内代码或转义文本中的 `![alt](data:image/...)` 当成真实图片，删除 alt 正文和原始语法。

现在普通消息与工具结果共用代码边界识别，保留反引号/波浪线围栏、不同长度及跨行的行内代码、转义、缩进代码和常见引用/列表容器中的原始文本。代码之外的图片仍按顺序转换为可用媒体；同一消息可以同时保留代码示例与真实图片。未增加依赖，也不声称实现了完整 CommonMark 解析器。

实现：[共享媒体提取](../../src-tauri/src/proxy/mappers/common_utils.rs)、[Claude 文本入口](../../src-tauri/src/proxy/mappers/claude/request.rs)。

### 3. Claude 结构化输出约束到达上游

原先 `output_config.format` 在 typed request 解析时消失，合法 JSON Schema 请求成为无格式约束的文本生成；模型变体路由还会整体清空 `output_config`。

现在保留 `format.type=json_schema` 及其 schema，通过已有共享 schema 清理器写入 `responseMimeType:application/json` 和 `responseSchema`。资源预算错误沿现有 `Result` 边界返回。变体路由仅消费 reasoning effort，保留独立的输出格式；直接变体与已路由模型两条路径均有回归。这与 Chat 使用相同的受支持 schema 子集，不承诺保留所有 JSON Schema 关键字。

实现：[Claude 输出配置类型](../../src-tauri/src/proxy/mappers/claude/models.rs)、[请求生成配置](../../src-tauri/src/proxy/mappers/claude/request.rs)、[变体路由](../../src-tauri/src/proxy/handlers/claude.rs)、[共享 schema 清理](../../src-tauri/src/proxy/common/json_schema.rs)。

### 4. legacy completions 区分失败、完成与截断

纯 `prompt` 的 `/v1/completions` 流式请求仍使用 legacy 适配器。原先 `OTHER`、`NO_IMAGE` 等模型失败会被映射成 `stop`；反过来，已经收到 STOP/MAX_TOKENS 后的连接重置又会追加要求重试的错误。

现在 legacy 接入共享生命周期的完成状态与短暂尾部收集，保留正常完成、长度上限、拒绝终态和后续稀疏 usage。模型执行失败使用共享错误分类，保留正文但不伪造成功。无 finishReason 的纯文本在 EOF、`[DONE]` 或 RST 后以 `length` 收尾；已观察到未完成工具调用时返回 `upstream_interrupted`，不授予执行完成状态。合成中断标记及 `FINISH_REASON_UNSPECIFIED` 不作为完成证据。

结构化上游错误仍走独立错误终结路径，不能被转换成正常完成或可恢复文本截断；legacy 终结只发送一次 `[DONE]`。

实现：[legacy 流式适配器](../../src-tauri/src/proxy/mappers/openai/streaming.rs)、[共享流生命周期](../../src-tauri/src/proxy/pipeline/stream_lifecycle.rs)、[实际 completions 入口](../../src-tauri/src/proxy/handlers/openai.rs)。

### 5. legacy completions 保留生成媒体

原先 legacy 仅处理 `part.text`，生成图片或文件时可能返回空 `text` 加 `stop`，丢掉主体结果。

现在复用共享 `render_part` 和 `MediaTracker`，把图片、文件链接及供应商文件引用保留在 legacy 文本字段中。相同帧中的重复媒体保留其数量，后续累计副本去重；媒体正确计为可见内容。实际 mapper/collector 对照确认文本、媒体及顺序与非流式路径使用的 Chat content 一致。

实现：[legacy 流式适配器](../../src-tauri/src/proxy/mappers/openai/streaming.rs)、[共享响应媒体渲染](../../src-tauri/src/proxy/common/response_media.rs)。图片模型请求 mapper 已验证会生成 `imageConfig` 和图片生成请求类型；最终 handler 内联的 Chat→legacy JSON 投影仅作静态确认，没有声称执行完整线上 HTTP 链路。

### 6. Responses 保留已经结束的模型拒绝

原先 Responses 仅将 STOP/MAX_TOKENS 和全提示词拒绝纳入尾部收集，候选 SAFETY/RECITATION/IMAGE_RECITATION 后的 RST 会把拒绝改成 `response.failed` 和 `stream_error`。

现在正常完成、长度上限及共享拒绝类别使用同一终态判断。候选拒绝有无可见解释都保留 `response.incomplete`、`content_filter` 和 `error:null`，连接重置不撤销模型已给出的终态，尾部 usage 仍可汇总。

共享生命周期还会在判断 transport Err 之前解析缓冲中的完整 JSON，覆盖首个或空拒绝终态没有尾随换行的情形。完整结构化 503 仍先由共享错误检测器处理；新增五条文本适配路径对照确认它不会因之前已有 STOP 或拒绝而被吞掉。

实现：[Responses 流式适配器](../../src-tauri/src/proxy/mappers/openai/streaming.rs)、[共享流生命周期](../../src-tauri/src/proxy/pipeline/stream_lifecycle.rs)。

## 独立复审闭环

请求与响应实现者互审，主审复核集成边界。新增回归实际发现 Claude 字符串入口先 `trim` 会删除四空格代码缩进，使共享识别器再次把源码提升为图片；已移除预先裁剪，保留空白检查，并恢复完整 mapper 的原文精确断言。

主审补充的引用和列表容器边界也已关闭：引用中的波浪线围栏保留为代码，列表内真正的图片继续转换为媒体，包括有序、嵌套、引用以及四空格缩进的列表场景。识别时处理容器前缀，保留文本时仍引用原始字节区间。

响应侧复审确认终态、媒体渲染、拒绝收尾及 RST 前缓冲解析一致；模型失败、结构化错误、可用文本截断与未完成工具仍可区分。本轮范围内无剩余已确认的阻断项。

## 验证与限制

唯一 Cargo runner 在真实工作树执行目标模块测试，最终 **202 passed / 0 failed / 1848 filtered**，执行 7.14 秒。其中包含 **17 项本轮永久回归：8 项请求、9 项响应**；17 已包含在 202 中，不相加。

永久回归入口：

- [请求历史测试](../../src-tauri/src/proxy/handlers/openai_request_history_tests.rs)：客户端搜索工具、格式约束、媒体源码与正常图片对照。
- [Claude variant tests](../../src-tauri/src/proxy/handlers/claude.rs)：effort 消费后仍保留结构化输出格式。
- [响应完成测试](../../src-tauri/src/proxy/mappers/openai/response_completion_tests.rs)：legacy 模型失败/截断/媒体/usage、Responses 拒绝、无换行结构化错误及 pending 连接。
- [既有流式测试](../../src-tauri/src/proxy/mappers/openai/streaming.rs)和[生命周期测试](../../src-tauri/src/proxy/pipeline/stream_lifecycle.rs)：关联路径回归；legacy 正文中断用例更新为保留 `length` 的契约。

从仓库根目录可重现最终测试范围：

```sh
ABV_DATA_DIR=/tmp/am-r34-runtime \
cargo test --manifest-path src-tauri/Cargo.toml --locked --lib -- \
  r34_ proxy::handlers::openai::request_history_tests \
  proxy::handlers::claude::variant_tests proxy::mappers::claude::request \
  proxy::mappers::common_utils proxy::mappers::openai::streaming \
  proxy::pipeline::stream_lifecycle \
  --skip server_authoritative_ignores_client_reasoning_content_and_budget \
  --skip test_openai_responses_api_vs_chat_api_thinking_and_signature \
  --test-threads=1
```

上面两项 reasoning 断言已在前轮旧精确基线复现失败，本轮沿用排除范围，没有修改其业务语义。既有编译警告仍存在；本报告不宣称全仓测试全绿。

所有验证使用本地夹具、生产 mapper、collector、observer 和生命周期；未调用真实供应商或进行模型采样。legacy 继续使用既有第一候选行为，未新增原生工具输出协议；普通 Chat 多候选 STOP+pending 不在此次修改范围。修改文件定向 rustfmt 和 `git diff --check` 通过；遵循仓库规则，未执行发布专用全仓检查、clippy、前端或发布构建。
