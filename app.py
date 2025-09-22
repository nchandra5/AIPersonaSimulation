import os
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from services.openai_client import get_openai_client
from services.persona_builder import build_persona_profile


# Load environment variables from .env if present
load_dotenv()


APP_TITLE = "Persona Sim"


def initialize_session_state() -> None:
    if "persona_profile" not in st.session_state:
        st.session_state["persona_profile"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "persona_name" not in st.session_state:
        st.session_state["persona_name"] = ""


def sidebar_persona_form() -> None:
    st.sidebar.header("Create / Update Persona")
    with st.sidebar.form("persona_form", clear_on_submit=False):
        full_name = st.text_input("Full name", key="input_full_name", placeholder="e.g., Jane Doe")
        linkedin_url = st.text_input("LinkedIn URL", key="input_linkedin", placeholder="https://www.linkedin.com/in/...")
        x_url = st.text_input("X (Twitter) URL", key="input_x", placeholder="https://x.com/...")
        additional_info = st.text_area(
            "Additional context (optional)",
            key="input_additional",
            height=140,
            placeholder="Any public details, topics, or constraints you want emphasized.",
        )
        submitted = st.form_submit_button("Create persona")

    if submitted:
        if not full_name and not (linkedin_url or x_url or additional_info):
            st.sidebar.warning("Please provide at least a name or one source link or additional info.")
            return

        with st.sidebar.status("Researching and synthesizing persona...", expanded=True) as status:
            try:
                client = get_openai_client()
                profile, redacted_name = build_persona_profile(
                    client=client,
                    full_name=full_name.strip(),
                    linkedin_url=linkedin_url.strip(),
                    x_url=x_url.strip(),
                    additional_info=additional_info.strip(),
                )
                st.session_state["persona_profile"] = profile
                st.session_state["persona_name"] = redacted_name or full_name or "Persona"
                st.session_state["messages"] = []
                status.update(label="Persona created successfully.", state="complete")
            except Exception as e:
                status.update(label="Failed to create persona.", state="error")
                st.sidebar.error(f"Error: {e}")


def render_chat_panel() -> None:
    st.title(APP_TITLE)

    if st.session_state.get("persona_profile") is None:
        st.info("Create a persona from the sidebar to begin chatting.")
        return

    persona_name = st.session_state.get("persona_name") or "Persona"
    st.subheader(f"Chatting with: {persona_name}")

    # Show persona profile expander
    with st.expander("View persona profile", expanded=False):
        st.markdown(st.session_state["persona_profile"]) 

    # Show conversation history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(placeholder=f"Talk with {persona_name}...")
    if user_input:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response conditioned on persona
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    assistant_reply = generate_persona_response(
                        st.session_state["messages"],
                        st.session_state["persona_profile"],
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
                st.markdown(assistant_reply)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})


def generate_persona_response(messages: List[Dict[str, str]], persona_profile: str) -> str:
    client = get_openai_client()

    developer_instructions = (
        "You are simulating a hypothetical person based on a synthesized persona profile. \n"
        "Goals: (1) embody their communication style, (2) ground content in their background, roles, "
        "and expertise, and (3) align behavior with an inferred personality type from the profile.\n\n"
        "Requirements:\n"
        "- Do not include or reveal the individual’s explicit full name or non-public identifiers.\n"
        "- Derive an implicit personality type (e.g., analytical, visionary, pragmatic, educator, "
        "operator, contrarian, empathetic leader). Calibrate tone, pacing, and word choice to match it.\n"
        "- Prioritize accuracy and faithfulness to the persona’s documented background/experience. "
        "If uncertain, state uncertainty plainly and prefer general principles consistent with their expertise.\n"
        "- Keep responses practical and high-signal. Use examples or frameworks the persona would plausibly use.\n"
        "- Avoid speculation beyond public, plausible inference. Do not fabricate private facts.\n"
        "- Refuse requests for sensitive personal data. Offer safe alternatives or high-level guidance.\n"
        "- Remain consistent in voice over the session. If the user requests a different style, adapt while "
        "staying anchored to the persona’s core traits.\n\n"
        "Output style: concise, direct, and aligned with the persona’s voice; cite uncertainty; avoid doxxing."
    )
    
    # Build the chat inputs: developer instructions + persona profile + conversation
    input_messages: List[Dict[str, Any]] = [
        {"role": "developer", "content": developer_instructions},
        {
            "role": "developer",
            "content": (
                "Persona Profile (redacted):\n\n" + persona_profile +
                "\n\nOperate as this hypothetical person in style and perspective."
            ),
        },
    ]

    # Append conversation history
    for msg in messages:
        input_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL_CHAT", "gpt-5"),
        reasoning={"effort": os.getenv("OPENAI_REASONING_EFFORT", "low")},
        input=input_messages,
    )

    return getattr(response, "output_text", "") or ""


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    initialize_session_state()
    sidebar_persona_form()
    render_chat_panel()


if __name__ == "__main__":
    main()


