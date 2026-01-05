"""
Gradio UI application for the Sidekick Personal Co-Worker.

This module provides the web interface for interacting with the Sidekick agent,
allowing users to submit tasks and success criteria.
"""

import gradio as gr
from sidekick import Sidekick


async def setup():
    """Initialize and set up a new Sidekick instance."""
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick


async def process_message(sidekick, message, success_criteria, history):
    """Process a user message through the Sidekick agent."""
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick


async def reset():
    """Reset the UI and create a new Sidekick instance."""
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick


def free_resources(sidekick):
    """Clean up resources when Sidekick instance is deleted."""
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


# ============================================================================
# Gradio UI Setup
# ============================================================================

with gr.Blocks(
    title="Sidekick",
    theme=gr.themes.Default(primary_hue="emerald")
) as ui:
    gr.Markdown("## Sidekick Personal Co-Worker")
    sidekick = gr.State(delete_callback=free_resources)

    # Chat interface
    with gr.Row():
        chatbot = gr.Chatbot(
            label="Sidekick",
            height=300,
            type="messages"
        )

    # Input fields
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(
                show_label=False,
                placeholder="Your request to the Sidekick"
            )
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False,
                placeholder="What are your success critiera?"
            )

    # Control buttons
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # Event handlers
    ui.load(setup, [], [sidekick])
    
    message.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick]
    )
    
    success_criteria.submit(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick]
    )
    
    go_button.click(
        process_message,
        [sidekick, message, success_criteria, chatbot],
        [chatbot, sidekick]
    )
    
    reset_button.click(
        reset,
        [],
        [message, success_criteria, chatbot, sidekick]
    )


if __name__ == "__main__":
    ui.launch(inbrowser=True)
