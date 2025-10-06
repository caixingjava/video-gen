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
  - 讯飞 TTS 负责配音合成，阿里云 DashScope 音乐生成（text-to-music）提供符合中国风格的背景音乐，Freesound 提供环境音效。
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
   - **OpenAI**：在 <https://platform.openai.com/account/api-keys> 申请 API Key，用于 GPT-4o 系列与 DALL·E 3 调用。
  - **讯飞 TTS**：使用科大讯飞“在线语音合成（Text-to-Speech WebAPI v2）”能力，在 <https://www.xfyun.cn/services/online_tts> 创建应用后获取 AppID、APIKey、APISecret。
  - **阿里云 DashScope 音乐生成**：在 <https://dashscope.aliyuncs.com/> 注册并开通「音频生成·音乐创作」能力，申请 API Key。
   - **Freesound**：登录 <https://freesound.org/apiv2/app/> 创建应用以获得个人 Token。

3. 编辑 `config/services.toml` 将上述凭证写入对应字段。若更倾向使用环境变量，也可以导出：

```bash
export OPENAI_API_KEY=sk-...
export XUNFEI_APP_ID=...
export XUNFEI_API_KEY=...
export XUNFEI_API_SECRET=...
export DASHSCOPE_API_KEY=...
export DASHSCOPE_MUSIC_MODEL=text-to-music-001
export DASHSCOPE_MUSIC_STYLE="中国古风"
export FREESOUND_API_KEY=...
```

未填写或缺失凭证时，系统会自动回退到内置的 Dummy Agent，便于在无真实账号的情况下进行流程验证。

### 4. 启动项目并提交关键词

1. 在任意终端中确认 `python --version` 仍然返回 3.9.10（如若不是，请重新调整 PATH 或改用 3.9.10 的可执行文件）。
2. 启动 FastAPI 服务：

```bash
python -m uvicorn video_gen.api.server:app --reload
```

   - 若在 Windows 上使用 `py` 启动，请改为 `py -3.9 -m uvicorn ...`。
3. 在另一个终端中（同样确保使用 Python 3.9.10）调用接口，输入目标人物姓名触发生成：

```bash
curl -X POST http://127.0.0.1:8000/tasks -H 'Content-Type: application/json' \
  -d '{"persona": "诸葛亮"}'
```

接口会返回剧本、分镜、素材提示词、运镜方案和音视频资源路径。凭证配置完整时将实际调用外部服务生成素材；缺省时会返回 Dummy 结果。
