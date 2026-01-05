import os
from dotenv import load_dotenv
load_dotenv(override=True)

import gradio as gr
from sidekick_tool import create_calendar_event, list_upcoming_events
from sidekick import Sidekick

async def setup():
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick

async def process_message(sidekick, message, success_criteria, history):
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick

async def reset():
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick


def free_resources(sidekick):
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")

# Gradio UI
with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Sidekick Personal Co-Worker")
    sidekick = gr.State(delete_callback=free_resources)

    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # Calendar Accordion
    with gr.Accordion("📆 Calendar", open=False):
        cal_summary     = gr.Textbox(label="Event Title")
        # Date/time inputs for user-friendly selection. Use text inputs for
        # dates (YYYY-MM-DD) because older Gradio versions may not expose a
        # Date component.
        cal_start_date  = gr.Textbox(label="Start date", placeholder="YYYY-MM-DD")
        cal_start_time  = gr.Textbox(label="Start time (HH:MM, 24h)", placeholder="15:00")
        cal_end_date    = gr.Textbox(label="End date", placeholder="YYYY-MM-DD")
        cal_end_time    = gr.Textbox(label="End time (HH:MM, 24h)", placeholder="16:00")

        # Populate timezone choices: try to use the system zoneinfo list, but
        # fall back to a small curated set if zoneinfo isn't available.
        try:
            from zoneinfo import available_timezones
            tz_choices = sorted(list(available_timezones()))
        except Exception:
            tz_choices = ["UTC", "America/Los_Angeles", "Europe/London", "Asia/Kolkata", "America/New_York"]

        # Try to auto-detect the local timezone; fall back to IST (Asia/Kolkata)
        try:
            from datetime import datetime as _dt
            local_tz_obj = _dt.now().astimezone().tzinfo
            default_tz = getattr(local_tz_obj, "key", None) or getattr(local_tz_obj, "zone", None) or str(local_tz_obj)
            if default_tz not in tz_choices:
                default_tz = "Asia/Kolkata"
        except Exception:
            default_tz = "Asia/Kolkata"

        tz_dropdown     = gr.Dropdown(choices=tz_choices, value=default_tz, label="Timezone")

        # Hidden RFC3339 fields that will be auto-filled from the pickers
        cal_start       = gr.Textbox(label="Start (RFC3339)", placeholder="2025-05-20T15:00:00", visible=False)
        cal_end         = gr.Textbox(label="End   (RFC3339)", placeholder="2025-05-20T16:00:00", visible=False)
        cal_description = gr.Textbox(label="Description (optional)")
        add_event_btn   = gr.Button("Add Event")
        list_events_btn = gr.Button("List Upcoming Events")
        cal_output      = gr.Textbox(label="Calendar Output", interactive=False)

    # Bind main functions
    ui.load(setup, [], [sidekick])
    message.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    success_criteria.submit(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    go_button.click(process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick])
    reset_button.click(reset, [], [message, success_criteria, chatbot, sidekick])

    # Bind calendar tools
    def prepare_datetimes(start_date, start_time, end_date, end_time):
        """Combine date and time strings into naive RFC3339-like datetimes (no offset).

        Returns (start_iso, end_iso, message). message is empty on success,
        otherwise contains a user-visible validation message.
        """
        def make_iso(d, t):
            if not d or not t:
                return ""
            # Ensure seconds are present
            t_full = t if len(t.split(':')) == 3 else f"{t}:00"
            return f"{d}T{t_full}"

        start_iso = make_iso(start_date, start_time)
        end_iso = make_iso(end_date, end_time)

        # Validate ordering if both are present
        msg = ""
        if start_iso and end_iso:
            try:
                from datetime import datetime as _dt
                s = _dt.fromisoformat(start_iso)
                e = _dt.fromisoformat(end_iso)
                if e <= s:
                    msg = "Error: End must be after start. Please correct the dates/times."
            except Exception:
                msg = "Error parsing date/time. Ensure times are HH:MM or HH:MM:SS and dates are YYYY-MM-DD."

        return start_iso, end_iso, msg

    # Auto-run prepare_datetimes whenever any of the date/time inputs change
    for comp in (cal_start_date, cal_start_time, cal_end_date, cal_end_time):
        comp.change(fn=prepare_datetimes, inputs=[cal_start_date, cal_start_time, cal_end_date, cal_end_time], outputs=[cal_start, cal_end, cal_output])

    def create_event_with_validation(summary, start_iso, end_iso, desc, tz):
        # Basic presence check
        if not summary:
            return "Error: Please provide an event title."
        if not start_iso or not end_iso:
            return "Error: Start and end datetimes must be prepared (choose dates and times)."

        # Validate ordering using timezone-aware comparison if possible
        try:
            from datetime import datetime as _dt
            try:
                from zoneinfo import ZoneInfo
                s = _dt.fromisoformat(start_iso).replace(tzinfo=ZoneInfo(tz))
                e = _dt.fromisoformat(end_iso).replace(tzinfo=ZoneInfo(tz))
            except Exception:
                s = _dt.fromisoformat(start_iso)
                e = _dt.fromisoformat(end_iso)
            if e <= s:
                return "Error: End must be after start. Please correct the inputs."
        except Exception:
            return "Error validating date/time values."

        return create_calendar_event(summary, start_iso, end_iso, desc, os.getenv("GOOGLE_CALENDAR_ID", "primary"), tz)

    add_event_btn.click(
        fn=create_event_with_validation,
        inputs=[cal_summary, cal_start, cal_end, cal_description, tz_dropdown],
        outputs=cal_output,
    )
    list_events_btn.click(
        fn=lambda: list_upcoming_events(os.getenv("GOOGLE_CALENDAR_ID", "primary")),
        outputs=cal_output
    )

ui.launch(inbrowser=True)