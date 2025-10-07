# video-gen

这是一个 AI 小视频生成工具的雏形实现，围绕六步大模型工作流完成架构搭建：

1. 剧本生成（含时间线）
2. 画面策划与分镜
3. 视觉素材生成
4. 运镜设计
5. 时间线/分镜视频合成
6. 配音并输出终版视频

当前仓库提供了以下能力：

- 基于 dataclass 的任务上下文与数据契约定义。
- 可串行执行的工作流编排器以及六类 Agent 接口。
- 一组确定性的 Dummy Agent，便于端到端跑通流程和后续替换为真实大模型调用。
- **生产级外部服务集成**：
  - 剧本/分镜/运镜/时间线全部使用 OpenAI 大模型（含 DALL·E 3 图像生成）。
  - 讯飞 TTS 负责配音合成，阿里云 DashScope 音乐生成（text-to-music）输出中国风格的配乐，并通过 DashScope 音频生成能力补充环境音效。
- FastAPI 原型接口，可创建任务并查询生成结果。

## 快速开始

按“确认系统 Python → 安装依赖 → 配置服务凭证 → 启动接口并提交任务”的顺序完成本地部署。

### 1. 确认系统 Python 版本

- 本项目面向 **Python 3.9.10**，请在终端执行 `python --version`（或 Windows 上使用 `py -3.9 --version`）确认默认解释器就是该版本。
- 若命令返回的不是 3.9.10，请根据操作系统调整 PATH 或直接使用对应的绝对路径，例如：
  - macOS/Linux：`/usr/bin/python3.9 --version`
  - Windows：`C:\Python39\python.exe --version`
- 后续所有命令都默认使用上述系统级 Python，可根据需要替换为实际可执行文件。若希望保持系统环境整洁，也可以使用 `python -m venv` 创建虚拟环境，但并非必需。

### 2. 安装项目依赖

在确认 `python` 指向 3.9.10 之后，安装/升级 pip 并以可编辑模式安装当前项目：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

如果你的系统同时存在多个 Python 版本，请显式调用 3.9.10 对应的可执行文件或在 Windows 上使用 `py -3.9 -m pip ...`。

### 3. 配置真实服务凭证

1. 复制示例配置文件：

```bash
cp config/services.example.toml config/services.toml
```

2. 按需填写以下字段（示例文件中已标注申请地址）：
    - **OpenAI**：在 <https://platform.openai.com/account/api-keys> 申请 API Key，用于 GPT-4o 系列与 DALL·E 3 调用。若你的机器需要通过本地代理访问 OpenAI，请直接在 `[openai]` 下把 `proxy` 改成自己的代理地址（例如 `http://127.0.0.1:7890`），这样工作流里所有 OpenAI 请求都会自动走这个本地代理；其他服务保持默认配置即可，它们不会再继承系统代理。
    - **讯飞 TTS**：使用科大讯飞“在线语音合成（Text-to-Speech WebAPI v2）”能力，在 <https://www.xfyun.cn/services/online_tts> 创建应用后获取 AppID、APIKey、APISecret。
    - **阿里云 DashScope 音乐生成 / 环境音效**：在 <https://dashscope.aliyun.com/> 注册并开通音频生成能力，单一 API Key 既可用于背景音乐，也可用于生成环境音效。
    - **DeepSeek**（可选）：在 <https://platform.deepseek.com/> 申请 API Key，按需配置自定义 `base_url`、模型、温度，以及网络相关的 `timeout_seconds`、`trust_env`（是否继承系统代理）或 `verify`（证书校验/自签路径），用于人物传记初稿的 LLM 推理。
    - **豆包图像生成**（可选）：在 <https://www.volcengine.com/product/doubao> 申请火山引擎豆包 API Key，可指定独立 `base_url`、模型、负面提示词，以及 `timeout_seconds`、`trust_env`/`verify` 等网络字段，用于角色立绘生成。

3. 编辑 `config/services.toml` 将上述凭证写入对应字段。若更倾向使用环境变量，也可以导出：

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_PROXY=http://127.0.0.1:7890   # 如果要走本地代理，请改成自己的端口
export OPENAI_TRUST_ENV=false               # 默认关闭系统代理继承
export XUNFEI_APP_ID=...
export XUNFEI_API_KEY=...
export XUNFEI_API_SECRET=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_MUSIC_MODEL=text-to-music-001
export DASHSCOPE_MUSIC_STYLE="中国古风"
export DASHSCOPE_AMBIENCE_API_KEY=$DASHSCOPE_API_KEY  # 若环境音效使用独立密钥可替换此行
export DASHSCOPE_AMBIENCE_STYLE="中国场景环境音"
export DASHSCOPE_AMBIENCE_DURATION=45
export DEEPSEEK_API_KEY=sk-deepseek...
export DEEPSEEK_BASE_URL=""  # 可选，默认为官方 SaaS 地址
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_TEMPERATURE=0.3
export DEEPSEEK_TIMEOUT_SECONDS=60
export DEEPSEEK_TRUST_ENV=true         # 默认即为 true，显式写出便于在部分网络禁用代理
export DEEPSEEK_VERIFY=true            # 也可指定 false 或 CA 证书路径
export DOUBAO_API_KEY=sk-doubao...
export DOUBAO_BASE_URL=""  # 可选，默认为官方 SaaS 地址
export DOUBAO_MODEL="doubao-vision"
export DOUBAO_NEGATIVE_PROMPT=""
export DOUBAO_TIMEOUT_SECONDS=60
export DOUBAO_TRUST_ENV=true
export DOUBAO_VERIFY=true
export TEXT_GENERATION_PROVIDER=openai  # 可切换为 deepseek
export IMAGE_GENERATION_PROVIDER=openai  # 可切换为 doubao
```

未填写或缺失凭证时，系统会自动回退到内置的 Dummy Agent，便于在无真实账号的情况下进行流程验证。

> 💡 OpenAI 客户端可以单独通过 `proxy` 字段或 `OPENAI_PROXY` 环境变量指定本地代理；其他服务默认不走代理。若部署在需要通过企业代理访问外网的环境，可为 DeepSeek / 豆包等客户端继续保留 `trust_env=true`（示例配置即为默认值），必要时通过 `verify=false` 或提供内部 CA 证书路径避免握手超时。

如需切换工作流使用的服务，可在 `config/services.toml` 中的 `[text_generation]` 与 `[image_generation]` 模块调整 `provider` 字段：

```toml
[text_generation]
provider = "deepseek"  # 使用 DeepSeek 生成“人物一生介绍”等文本

[image_generation]
provider = "doubao"  # 使用豆包生成分镜图像
```

同样也可以通过环境变量 `TEXT_GENERATION_PROVIDER` 和 `IMAGE_GENERATION_PROVIDER` 动态切换；未显式配置时默认仍调用 OpenAI。

### 4. 启动项目并提交关键词

1. 在任意终端中确认 `python --version` 仍然返回 3.9.10（如若不是，请重新调整 PATH 或改用 3.9.10 的可执行文件）。
2. 启动 FastAPI 服务：

```bash
python -m uvicorn video_gen.api.server:app --reload
```

   - 若在 Windows 上使用 `py` 启动，请改为 `py -3.9 -m uvicorn ...`。
3. 打开浏览器访问 <http://127.0.0.1:8000/>，即可看到“关键词生成”页面：
   - 在输入框中键入想要生成的历史人物或形象（例如“诸葛亮”）。
   - 点击“开始生成”后，页面会依次显示工作流 6 个阶段的输出内容。
   - 全流程结束后，底部会出现一个预览区，展示最终视频的可播放链接。
   - 页面以轮询方式刷新任务状态，无需手动刷新即可看到每一步最新结果。
4. 如果更习惯通过命令行直接触发，也可以在另一个终端中（同样确保使用 Python 3.9.10）调用接口：

```bash
curl -X POST http://127.0.0.1:8000/tasks -H 'Content-Type: application/json' \
  -d '{"persona": "诸葛亮"}'
```

接口会返回剧本、分镜、素材提示词、运镜方案和音视频资源路径。凭证配置完整时将实际调用外部服务生成素材；缺省时会返回 Dummy 结果。
