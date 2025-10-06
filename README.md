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

## 快速开始

按“下载并安装 Miniconda → 准备项目环境 → 配置服务凭证 → 启动接口并提交任务”的顺序完成本地部署。

### 1. 下载 Miniconda

- 访问官方发布页：<https://docs.conda.io/en/latest/miniconda.html>
- 根据本机操作系统选择对应的安装包（Windows/macOS/Linux），推荐下载最新的 64 位安装程序。

### 2. 安装 Miniconda 并在项目目录内创建虚拟环境

1. 按官网指引运行安装程序：
   - **Windows**：双击 `.exe`，在向导中勾选“Add Miniconda to my PATH”（或在后续步骤手动添加）。
   - **macOS/Linux**：为 `.sh` 文件增加执行权限（`chmod +x`），随后运行 `./Miniconda3-latest-*.sh` 并跟随提示完成安装。
2. 安装完成后重新打开终端（或在 Windows 使用“Miniconda Prompt”），进入本项目目录，然后执行以下命令在仓库根目录内创建独立环境：

```bash
# 进入项目根目录，例如：
cd /path/to/video-gen

# 使用 --prefix 将环境固定到当前仓库的 .conda 目录
conda create --yes --prefix ./.conda python=3.12
conda activate ./\.conda
```

> `conda activate ./\.conda` 会将解释器切换到项目内部的 `.conda` 文件夹，避免与全局环境互相影响。若需删除环境，直接移除该目录即可（`rm -rf .conda`）。

3. 在激活的项目环境中安装依赖：

```bash
pip install --upgrade pip
pip install -e .
```

### 3. 配置真实服务凭证

1. 复制示例配置文件：

```bash
cp config/services.example.toml config/services.toml
```

2. 按需填写以下字段（示例文件中已标注申请地址）：
   - **OpenAI**：在 <https://platform.openai.com/account/api-keys> 申请 API Key，用于 GPT-4o 系列与 DALL·E 3 调用。
   - **讯飞 TTS**：在科大讯飞开放平台 <https://www.xfyun.cn/services/online_tts> 创建应用后获取 AppID、APIKey、APISecret。
   - **Mubert**：在 <https://mubert.com/render/pricing> 注册并申请 API Key。
   - **Freesound**：登录 <https://freesound.org/apiv2/app/> 创建应用以获得个人 Token。

3. 编辑 `config/services.toml` 将上述凭证写入对应字段。若更倾向使用环境变量，也可以导出：

```bash
export OPENAI_API_KEY=sk-...
export XUNFEI_APP_ID=...
export XUNFEI_API_KEY=...
export XUNFEI_API_SECRET=...
export MUBERT_API_KEY=...
export FREESOUND_API_KEY=...
```

未填写或缺失凭证时，系统会自动回退到内置的 Dummy Agent，便于在无真实账号的情况下进行流程验证。

### 4. 启动项目并提交关键词

1. 确保仍然处于项目专用 Conda 环境：`conda activate ./\.conda`。
2. 启动 FastAPI 服务：

```bash
uvicorn video_gen.api.server:app --reload
```

3. 在另一个终端内（同样激活环境）调用接口，输入目标人物姓名触发生成：

```bash
curl -X POST http://127.0.0.1:8000/tasks -H 'Content-Type: application/json' \
  -d '{"persona": "诸葛亮"}'
```

接口会返回剧本、分镜、素材提示词、运镜方案和音视频资源路径。凭证配置完整时将实际调用外部服务生成素材；缺省时会返回 Dummy 结果。
