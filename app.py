import gradio as gr

from paper_chatbot.config import Settings
from paper_chatbot.rag import RAGChatbot


def build_app():
    settings = Settings.from_env()
    rag = None
    startup_error = None
    try:
        rag = RAGChatbot.from_settings(settings)
    except Exception as exc:  # noqa: BLE001
        startup_error = str(exc)

    with gr.Blocks(title="Paper Chatbot") as demo:
        gr.Markdown("# Paper Chatbot")
        if startup_error:
            gr.Markdown(
                "**Startup error:** {error}\n\n"
                "Build the corpus and Chroma index, then restart the app.".format(
                    error=startup_error
                )
            )

        with gr.Row():
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=settings.top_k,
                step=1,
                label="Top K",
            )

        chatbot = gr.Chatbot(height=520)
        gr.Markdown("## Sources")
        sources = gr.Markdown()
        msg = gr.Textbox(placeholder="Ask a question about the papers...", show_label=False)

        with gr.Row():
            send = gr.Button("Send")
            clear = gr.Button("Clear")

        def _messages_to_pairs(messages):
            pairs = []
            pending_user = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    pending_user = content
                elif role == "assistant":
                    if pending_user is None:
                        pairs.append(("", content))
                    else:
                        pairs.append((pending_user, content))
                        pending_user = None
            return pairs

        def respond(message, history, k):
            history = history or []
            if not message:
                return history, "", ""
            if rag is None:
                error_msg = "Index is not available. Please build the corpus and Chroma index."
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_msg},
                ]
                return history, "", ""

            pair_history = _messages_to_pairs(history)
            answer, sources_text = rag.answer(message, pair_history, top_k=int(k))
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, "", sources_text

        send.click(respond, inputs=[msg, chatbot, top_k], outputs=[chatbot, msg, sources])
        msg.submit(respond, inputs=[msg, chatbot, top_k], outputs=[chatbot, msg, sources])

        clear.click(lambda: ([], "", ""), outputs=[chatbot, msg, sources])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
