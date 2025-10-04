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
  - 讯飞 TTS 负责配音合成，Mubert 自动生成背景音乐，Freesound 提供环境音效。
- FastAPI 原型接口，可创建任务并查询生成结果。
- Pytest 用例，用于验证工作流串联是否符合预期。

## 快速开始

```bash
pip install -e .[dev]
pytest
uvicorn video_gen.api.server:app --reload
```

### 配置真实服务

仓库提供了 `config/services.example.toml` 示例文件，包含 OpenAI、讯飞 TTS、Mubert、Freesound 的所需凭证字段。复制并填写实际值：

```bash
cp config/services.example.toml config/services.toml
# 编辑 config/services.toml 填入真实的 API Key/AppID 等信息
```

启动服务时会自动读取 `config/services.toml`。也可通过环境变量覆盖，例如：

```bash
export OPENAI_API_KEY=sk-...
export XUNFEI_APP_ID=...
export XUNFEI_API_KEY=...
export XUNFEI_API_SECRET=...
export MUBERT_API_KEY=...
export FREESOUND_API_KEY=...
```

当所有凭证均可用时，任务管理器会自动切换到生产版 Agent，调用真实的大模型、配音与音效服务；否则保持使用 Dummy Agent 以便本地开发。

启动服务后，可通过以下请求体验同步执行的示例流程：

```bash
curl -X POST http://localhost:8000/tasks -H 'Content-Type: application/json' \
  -d '{"persona": "诸葛亮"}'
```

返回的 JSON 中会包含脚本、分镜、素材提示词、运镜方案以及虚拟的成片地址。实际项目可将 Dummy Agent 替换成对接大模型、图像/视频生成与 TTS 服务的实现。
