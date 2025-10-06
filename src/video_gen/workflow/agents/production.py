"""Production-ready agent implementations backed by external services."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import (
    CameraInstruction,
    FinalAssets,
    ScriptSection,
    StoryboardShot,
    TaskContext,
    TimelineCue,
    TimelineEntry,
    TimelineLayer,
    VisualAsset,
)
from .base import AssetAgent, CameraAgent, ScriptAgent, SynthesisAgent, TimelineAgent, VisualPlannerAgent
from ...config import ServiceConfig
from ...providers import (
    DashscopeAmbienceClient,
    DashscopeMusicClient,
    DeepSeekWorkflowClient,
    DoubaoImageClient,
    OpenAIWorkflowClient,
    XunfeiTTSClient,
)


class OpenAIScriptAgent(ScriptAgent):
    def __init__(self, client: OpenAIWorkflowClient) -> None:
        self._client = client

    def run(self, context: TaskContext) -> List[ScriptSection]:
        return self._client.generate_script(context.persona).sections


class DeepSeekScriptAgent(ScriptAgent):
    def __init__(self, client: DeepSeekWorkflowClient) -> None:
        self._client = client

    def run(self, context: TaskContext) -> List[ScriptSection]:
        return self._client.generate_script(context.persona).sections


class OpenAIVisualPlannerAgent(VisualPlannerAgent):
    def __init__(self, client: OpenAIWorkflowClient) -> None:
        self._client = client

    def run(self, context: TaskContext, script: List[ScriptSection]) -> List[StoryboardShot]:
        return self._client.generate_storyboard(context.persona, script).shots


class OpenAIAssetAgent(AssetAgent):
    def __init__(self, client: OpenAIWorkflowClient) -> None:
        self._client = client

    def run(self, context: TaskContext, storyboard: List[StoryboardShot]) -> List[VisualAsset]:
        assets: List[VisualAsset] = []
        for shot in storyboard:
            prompt = (
                f"{context.persona} {shot.scene}. Historical authenticity, cinematic lighting, fine details."
            )
            asset = self._client.generate_dalle_image(shot.shot_id, prompt)
            assets.append(asset)
        return assets


class DoubaoAssetAgent(AssetAgent):
    def __init__(self, client: DoubaoImageClient) -> None:
        self._client = client

    def run(self, context: TaskContext, storyboard: List[StoryboardShot]) -> List[VisualAsset]:
        assets: List[VisualAsset] = []
        for shot in storyboard:
            prompt = (
                f"{context.persona} {shot.scene}. 中国风格，高清细节，纪录片质感。"
            )
            result = self._client.generate_image(shot.shot_id, prompt)
            assets.append(result.asset)
        return assets


class OpenAICameraAgent(CameraAgent):
    def __init__(self, client: OpenAIWorkflowClient) -> None:
        self._client = client

    def run(self, context: TaskContext, storyboard: List[StoryboardShot]) -> List[CameraInstruction]:
        return self._client.generate_camera_plan(storyboard).instructions


class OpenAITimelineAgent(TimelineAgent):
    def __init__(self, client: OpenAIWorkflowClient) -> None:
        self._client = client

    def run(
        self,
        context: TaskContext,
        storyboard: List[StoryboardShot],
        assets: List[VisualAsset],
        camera_plan: List[CameraInstruction],
    ) -> List[TimelineEntry]:
        return self._client.generate_timeline(storyboard, assets, camera_plan).entries


class ExternalSynthesisAgent(SynthesisAgent):
    def __init__(
        self,
        tts_client: XunfeiTTSClient,
        music_client: DashscopeMusicClient,
        ambience_client: DashscopeAmbienceClient,
        output_dir: Path,
    ) -> None:
        self._tts = tts_client
        self._music = music_client
        self._ambience = ambience_client
        self._output_dir = output_dir

    def _render_narration(self, context: TaskContext, timeline: List[TimelineEntry]) -> List[Path]:
        narration_files: List[Path] = []
        for index, entry in enumerate(timeline):
            for cue in entry.audio_cues:
                if cue.cue_type != "narration":
                    continue
                filename = self._output_dir / f"{context.task_id}_narration_{index:02d}.mp3"
                self._tts.synthesize(cue.reference, filename)
                narration_files.append(filename)
        return narration_files

    def run(self, context: TaskContext, timeline: List[TimelineEntry]) -> FinalAssets:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        narration_files = self._render_narration(context, timeline)
        bgm_path = self._music.generate_track(
            context.persona, self._output_dir / f"{context.task_id}_bgm.mp3"
        )
        ambience_path = self._ambience.generate_ambience(
            context.persona, self._output_dir / f"{context.task_id}_ambience.mp3"
        )

        video_placeholder = self._output_dir / f"{context.task_id}.mp4"
        if not video_placeholder.exists():
            video_placeholder.write_bytes(b"")

        metadata = {
            "narration_files": [str(path) for path in narration_files],
            "background_music": str(bgm_path),
            "ambience": str(ambience_path),
        }

        subtitles_path = self._output_dir / f"{context.task_id}.srt"
        if timeline:
            subtitles_lines: List[str] = []
            for entry_index, entry in enumerate(timeline, start=1):
                for cue in entry.audio_cues:
                    start_seconds = cue.start.total_seconds()
                    end_seconds = start_seconds + cue.duration.total_seconds()
                    start_ts = _format_timestamp(start_seconds)
                    end_ts = _format_timestamp(end_seconds)
                    subtitles_lines.extend(
                        [
                            str(entry_index),
                            f"{start_ts} --> {end_ts}",
                            cue.reference,
                            "",
                        ]
                    )
            subtitles_path.write_text("\n".join(subtitles_lines), encoding="utf-8")

        return FinalAssets(
            video_uri=str(video_placeholder),
            audio_uri=str(bgm_path),
            subtitles_uri=str(subtitles_path) if subtitles_path.exists() else None,
            metadata=metadata,
        )


def _format_timestamp(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def create_production_agents(config: ServiceConfig):
    """Factory that builds the production agents using configured services."""

    if (
        not config.openai
        or not config.xunfei
        or not config.dashscope_music
        or not config.dashscope_ambience
    ):
        missing = [
            name
            for name, value in {
                "openai": config.openai,
                "xunfei": config.xunfei,
                "dashscope_music": config.dashscope_music,
                "dashscope_ambience": config.dashscope_ambience,
            }.items()
            if value is None
        ]
        raise RuntimeError(
            "Production agents require OpenAI, Xunfei, and DashScope music/ambience settings. "
            f"Missing: {', '.join(missing)}"
        )

    openai_client = OpenAIWorkflowClient(config.openai)
    text_provider = (config.text_generation.provider or "openai").lower()
    if text_provider == "deepseek":
        if not config.deepseek:
            raise RuntimeError("DeepSeek provider selected but no deepseek settings supplied")
        deepseek_client = DeepSeekWorkflowClient(config.deepseek)
        script_agent = DeepSeekScriptAgent(deepseek_client)
    else:
        script_agent = OpenAIScriptAgent(openai_client)
    visual_agent = OpenAIVisualPlannerAgent(openai_client)
    image_provider = (config.image_generation.provider or "openai").lower()
    if image_provider == "doubao":
        if not config.doubao:
            raise RuntimeError("Doubao provider selected but no doubao settings supplied")
        doubao_client = DoubaoImageClient(config.doubao)
        asset_agent = DoubaoAssetAgent(doubao_client)
    else:
        asset_agent = OpenAIAssetAgent(openai_client)
    camera_agent = OpenAICameraAgent(openai_client)
    timeline_agent = OpenAITimelineAgent(openai_client)

    output_dir = Path(config.storage.output_dir)
    tts_client = XunfeiTTSClient(config.xunfei)
    music_client = DashscopeMusicClient(config.dashscope_music)
    ambience_client = DashscopeAmbienceClient(config.dashscope_ambience)
    synthesis_agent = ExternalSynthesisAgent(
        tts_client, music_client, ambience_client, output_dir
    )

    return (
        script_agent,
        visual_agent,
        asset_agent,
        camera_agent,
        timeline_agent,
        synthesis_agent,
    )


__all__ = [
    "OpenAIScriptAgent",
    "DeepSeekScriptAgent",
    "OpenAIVisualPlannerAgent",
    "OpenAIAssetAgent",
    "DoubaoAssetAgent",
    "OpenAICameraAgent",
    "OpenAITimelineAgent",
    "ExternalSynthesisAgent",
    "create_production_agents",
]
