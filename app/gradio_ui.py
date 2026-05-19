from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Optional, Tuple

import gradio as gr


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MultiAgentGradio")

_SYSTEM = None
_SYSTEM_LOCK = Lock()


def get_system():
    global _SYSTEM
    if _SYSTEM is None:
        from app import MultiAgentSystem

        _SYSTEM = MultiAgentSystem()
    return _SYSTEM


def transcribe_audio(
    audio_path: Optional[str],
    current_question: str,
    language: str,
) -> Tuple[str, str]:
    if not audio_path:
        return current_question, "Record audio first."

    try:
        import speech_recognition as sr
    except ImportError:
        return (
            current_question,
            "Install SpeechRecognition first: pip install SpeechRecognition",
        )

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        transcript = recognizer.recognize_google(audio, language=language).strip()
        merged = f"{current_question.strip()} {transcript}".strip()
        return merged, "Audio converted to text."
    except sr.UnknownValueError:
        return current_question, "I could not understand the audio."
    except sr.RequestError as exc:
        return current_question, f"Speech service error: {exc}"
    except Exception as exc:
        logger.exception("Audio transcription failed")
        return current_question, f"Audio transcription failed: {exc}"


def ask_agents(question: str) -> Tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "", "Type a question or transcribe microphone audio first."

    try:
        with _SYSTEM_LOCK:
            result = asyncio.run(get_system().run(question))

        answer = result.get("answer", "No answer returned.")
        score = result.get("score", 0.0)
        source = result.get("source", "unknown")
        details = f"Source: {source} | Score: {score}"
        return answer, details
    except Exception as exc:
        logger.exception("Agent request failed")
        return "", f"Agent request failed: {exc}"


def clear_all():
    return None, "", "", "Ready."


with gr.Blocks(title="Multi-Agent Research") as demo:
    gr.Markdown(
        """
        # Multi-Agent Research
        Ask with text, or record your voice and convert it to text before sending it to the agents.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Question",
                lines=5,
                placeholder="Type your question here, or record audio and click Transcribe.",
            )
            with gr.Row():
                ask_button = gr.Button("Ask agents", variant="primary")
                clear_button = gr.Button("Clear")

        with gr.Column(scale=1):
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="Microphone",
            )
            language = gr.Dropdown(
                choices=[
                    ("English", "en-US"),
                    ("Arabic", "ar-EG"),
                ],
                value="en-US",
                label="Speech language",
            )
            transcribe_button = gr.Button("Transcribe audio")
            status = gr.Textbox(label="Status", value="Ready.", interactive=False)

    answer = gr.Textbox(label="Answer", lines=12, interactive=False, buttons=["copy"])
    details = gr.Textbox(label="Details", interactive=False)

    gr.Examples(
        examples=[
            "What does Alice learn in the local documents?",
            "Summarize the most important local evidence.",
            "What should I know before researching agentic RAG?",
        ],
        inputs=question,
    )

    transcribe_button.click(
        fn=transcribe_audio,
        inputs=[audio, question, language],
        outputs=[question, status],
    )
    ask_button.click(fn=ask_agents, inputs=question, outputs=[answer, details])
    question.submit(fn=ask_agents, inputs=question, outputs=[answer, details])
    clear_button.click(fn=clear_all, outputs=[audio, question, answer, status])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
