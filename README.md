# video-gen

这是一个 AI 小视频生成工具的雏形实现，围绕六步大模型工作流完成架构搭建：

1. 剧本生成（含时间线）
2. 画面策划与分镜
3. 视觉素材生成
4. 运镜设计
5. 时间线/分镜视频合成
6. 配音并输出终版视频

当前仓库提供了以下能力：

- 基于 Pydantic 的任务上下文与数据契约定义。
- 可串行执行的工作流编排器以及六类 Agent 接口。
- 一组确定性的 Dummy Agent，便于端到端跑通流程和后续替换为真实大模型调用。
- FastAPI 原型接口，可创建任务并查询生成结果。
- Pytest 用例，用于验证工作流串联是否符合预期。

## 快速开始

```bash
pip install -e .[dev]
pytest
uvicorn video_gen.api.server:app --reload
```

启动服务后，可通过以下请求体验同步执行的示例流程：

```bash
curl -X POST http://localhost:8000/tasks -H 'Content-Type: application/json' \
  -d '{"persona": "诸葛亮"}'
```

返回的 JSON 中会包含脚本、分镜、素材提示词、运镜方案以及虚拟的成片地址。实际项目可将 Dummy Agent 替换成对接大模型、图像/视频生成与 TTS 服务的实现。
