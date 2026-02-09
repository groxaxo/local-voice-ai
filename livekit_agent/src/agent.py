import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    async def on_enter(self):
        # Generate a greeting when the agent joins the session.
        # Keep it uninterruptible so the client has time to calibrate AEC.
        self.session.generate_reply(allow_interruptions=False)

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.

        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """

        return f"The product of {number1} and {number2} is {number1 * number2}."


def _build_stt() -> openai.STT:
    """Return the STT instance based on STT_PROVIDER env var."""
    provider = os.getenv("STT_PROVIDER", "parakeet").lower()
    if provider == "whisper":
        return openai.STT(
            base_url="http://whisper:80/v1",
            model="Systran/faster-whisper-small",
            api_key="no-key-needed",
        )
    # default: parakeet
    return openai.STT(
        base_url="http://parakeet:8015/v1",
        model=os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"),
        api_key="no-key-needed",
    )


def _build_tts() -> openai.TTS:
    """Return the TTS instance based on TTS_PROVIDER env var."""
    provider = os.getenv("TTS_PROVIDER", "kokoro").lower()
    if provider == "soprano":
        return openai.TTS(
            base_url="http://soprano:8000/v1",
            model="soprano",
            voice="default",
            api_key="no-key-needed",
        )
    # default: kokoro
    return openai.TTS(
        base_url="http://kokoro:8880/v1",
        model="kokoro",
        voice="af_nova",
        api_key="no-key-needed",
    )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    llm_model = os.getenv("VLLM_MODEL_ALIAS", "gemma-3-27b")
    llm_base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")

    session = AgentSession(
        stt=_build_stt(),
        llm=openai.LLM(base_url=llm_base_url, model=llm_model, api_key="no-key-needed"),
        tts=_build_tts(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # Log metrics as they are emitted and collect total usage
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
