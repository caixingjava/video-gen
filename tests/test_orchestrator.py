from video_gen.workflow.agents.dummy import create_dummy_agents
from video_gen.workflow.models import TaskState
from video_gen.workflow.orchestrator import VideoGenerationOrchestrator


def test_orchestrator_runs_full_workflow():
    (
        script_agent,
        visual_planner,
        asset_agent,
        camera_agent,
        timeline_agent,
        synthesis_agent,
    ) = create_dummy_agents()
    orchestrator = VideoGenerationOrchestrator(
        script_agent,
        visual_planner,
        asset_agent,
        camera_agent,
        timeline_agent,
        synthesis_agent,
    )

    context = orchestrator.create_context("诸葛亮")
    result = orchestrator.run(context)

    assert result.state == TaskState.DELIVERED
    assert len(result.script) == 3
    assert len(result.storyboard) == 3
    assert len(result.assets) == 3
    assert len(result.camera_plan) == 3
    assert len(result.timeline) == 3
    assert result.final_assets is not None
    assert result.final_assets.video_uri.endswith("video.mp4")
