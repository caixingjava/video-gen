"""FastAPI server exposing the video generation workflow."""

from __future__ import annotations

import logging

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = object  # type: ignore
    HTMLResponse = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    class BaseModel:  # type: ignore
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict:
            return self.__dict__

        def dict(self, *args, **kwargs):  # noqa: D401
            """Compatibility alias for Pydantic"""

            return self.__dict__

from ..services.task_manager import TASK_MANAGER
from ..workflow.models import TaskContext


class CreateTaskRequest(BaseModel):
    persona: str


class TaskResponse(BaseModel):
    task: dict


INDEX_HTML = """<!DOCTYPE html>
<html lang=\"zh\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>历史人物视频生成器</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: 'Segoe UI', 'PingFang SC', system-ui, -apple-system, sans-serif;
        background: #f5f7fa;
        color: #222;
      }

      body {
        margin: 0;
        padding: 2rem;
        display: flex;
        justify-content: center;
      }

      main {
        width: min(900px, 100%);
        background: rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.15);
        padding: 2.5rem 3rem;
        backdrop-filter: blur(8px);
      }

      h1 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: clamp(1.8rem, 2.4vw, 2.6rem);
        color: #1f2937;
      }

      p.lead {
        margin-top: 0;
        color: #4b5563;
        line-height: 1.6;
      }

      form {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
      }

      label {
        flex: 1 0 160px;
        font-weight: 600;
        color: #111827;
      }

      input[type='text'] {
        flex: 3 1 280px;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        font-size: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }

      input[type='text']:focus {
        outline: none;
        border-color: #2563eb;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
      }

      button {
        flex: 0 0 auto;
        padding: 0.8rem 1.6rem;
        border: none;
        border-radius: 999px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.2s ease;
      }

      button:disabled {
        opacity: 0.6;
        cursor: wait;
        box-shadow: none;
      }

      button:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.35);
      }

      #status {
        margin-top: 0.5rem;
        font-size: 0.95rem;
        color: #2563eb;
      }

      section#results {
        margin-top: 2.5rem;
        display: grid;
        gap: 1.5rem;
      }

      .step-grid {
        display: grid;
        gap: 1rem;
      }

      .step-card {
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(255, 255, 255, 0.75);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
      }

      .step-card h4 {
        margin: 0 0 0.6rem;
        color: #0f172a;
      }

      .card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(244, 246, 252, 0.95));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(209, 213, 219, 0.6);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
      }

      .card h3 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        color: #111827;
      }

      pre {
        margin: 0;
        font-size: 0.9rem;
        background: rgba(15, 23, 42, 0.05);
        padding: 1rem;
        border-radius: 12px;
        overflow-x: auto;
      }

      video {
        width: 100%;
        border-radius: 12px;
        margin-top: 1rem;
        background: #000;
      }

      .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.12);
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 600;
      }

      @media (max-width: 640px) {
        body {
          padding: 1.5rem;
        }

        main {
          padding: 1.8rem;
        }

        form {
          flex-direction: column;
          gap: 0.75rem;
        }

        button {
          width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <h1>历史人物短视频演示</h1>
        <p class=\"lead\">输入关键词（例如“诸葛亮”），系统会依次完成剧本、分镜、素材、镜头设计、时间线与合成，最后展示一个示例视频链接。</p>
      </header>
      <form id=\"task-form\">
        <label for=\"persona\">历史人物 / 主题：</label>
        <input id=\"persona\" name=\"persona\" type=\"text\" placeholder=\"例如：诸葛亮\" required />
        <button type=\"submit\">开始生成</button>
      </form>
      <div id=\"status\"></div>
      <section id=\"results\" hidden>
        <div class=\"card\">
          <h3>任务状态</h3>
          <div><span class=\"tag\" id=\"state\">等待任务</span></div>
          <p id=\"task-id\"></p>
        </div>
        <div class=\"card\">
          <h3>阶段输出</h3>
          <div id=\"steps\" class=\"step-grid\"></div>
        </div>
        <div class=\"card\">
          <h3>最终视频预览</h3>
          <video id=\"video-preview\" controls preload=\"metadata\" hidden></video>
          <p id=\"video-link\">尚未生成视频链接</p>
        </div>
      </section>
    </main>
    <script>
      const form = document.getElementById('task-form');
      const personaInput = document.getElementById('persona');
      const statusBox = document.getElementById('status');
      const resultsSection = document.getElementById('results');
      const stepsContainer = document.getElementById('steps');
      const stateTag = document.getElementById('state');
      const taskIdLabel = document.getElementById('task-id');
      const videoElement = document.getElementById('video-preview');
      const videoLink = document.getElementById('video-link');

      const pipeline = [
        { key: 'script', label: '剧本阶段 (SCRIPTING)' },
        { key: 'storyboard', label: '分镜阶段 (VISUAL_PLANNING)' },
        { key: 'assets', label: '素材阶段 (ASSET_GENERATION)' },
        { key: 'camera_plan', label: '镜头设计 (CAMERA_DESIGN)' },
        { key: 'timeline', label: '时间线阶段 (TIMELINE_BUILD)' }
      ];

      const renderers = {
        default(container, value) {
          const pre = document.createElement('pre');
          if (value && ((Array.isArray(value) && value.length) || Object.keys(value || {}).length)) {
            pre.textContent = JSON.stringify(value, null, 2);
          } else {
            pre.textContent = '暂无数据';
          }
          container.appendChild(pre);
        },
        final_assets(value) {
          if (value && value.video_uri) {
            videoElement.hidden = false;
            videoElement.src = value.video_uri;
            videoElement.load();
            videoLink.innerHTML = `视频地址：<a href="${value.video_uri}" target="_blank" rel="noopener">${value.video_uri}</a>`;
          } else {
            videoElement.hidden = true;
            videoElement.removeAttribute('src');
            videoLink.textContent = '尚未生成视频链接';
          }
        }
      };

      async function createTask(persona) {
        const response = await fetch('/tasks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ persona })
        });
        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || '请求失败');
        }
        return response.json();
      }

      function renderTask(task) {
        resultsSection.hidden = false;
        stateTag.textContent = `当前状态：${task.state}`;
        taskIdLabel.textContent = `任务 ID：${task.task_id}`;
        stepsContainer.innerHTML = '';

        pipeline.forEach((step) => {
          const block = document.createElement('article');
          block.className = 'step-card';
          const title = document.createElement('h4');
          title.textContent = step.label;
          block.appendChild(title);
          renderers.default(block, task[step.key]);
          stepsContainer.appendChild(block);
        });

        renderers.final_assets(task.final_assets || null);
      }

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const persona = personaInput.value.trim();
        if (!persona) {
          statusBox.textContent = '请输入关键词';
          return;
        }

        form.querySelector('button').disabled = true;
        statusBox.textContent = '正在生成，请稍候…';

        try {
          const response = await createTask(persona);
          if (!response.task) {
            throw new Error('接口返回数据格式不正确');
          }
          renderTask(response.task);
          statusBox.textContent = '生成完成，以下为各阶段输出示例。';
        } catch (error) {
          console.error(error);
          statusBox.textContent = `生成失败：${error.message}`;
        } finally {
          form.querySelector('button').disabled = false;
        }
      });
    </script>
  </body>
</html>"""


LOGGER = logging.getLogger(__name__)


if FastAPI is not None:  # pragma: no branch - executed only when FastAPI is available
    app = FastAPI(title="Video Generation Workflow")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> str:  # pragma: no cover - integration path
        LOGGER.info("GET %s", request.url)
        return INDEX_HTML

    @app.post("/tasks", response_model=TaskResponse)
    def create_task(  # pragma: no cover - integration path
        request: Request, body: CreateTaskRequest
    ) -> TaskResponse:
        LOGGER.info("POST %s", request.url)
        context = TASK_MANAGER.start_task(body.persona)
        return TaskResponse(task=context.to_dict())

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(  # pragma: no cover - integration path
        request: Request, task_id: str
    ) -> TaskResponse:
        LOGGER.info("GET %s", request.url)
        context = TASK_MANAGER.get_task(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskResponse(task=context.to_dict())
else:  # pragma: no cover - executed in minimal environments
    app = None  # type: ignore
