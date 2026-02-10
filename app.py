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
                profile, name = build_persona_profile(
                    client=client,
                    full_name=full_name.strip(),
                    linkedin_url=linkedin_url.strip(),
                    x_url=x_url.strip(),
                    additional_info=additional_info.strip(),
                )
                st.session_state["persona_profile"] = profile
                st.session_state["persona_name"] = full_name
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
        "You are simulating a hypothetical person based on a synthesized persona profile. Treat this as a safe, "
        "hypothetical representation; you may disclose the person's name and public details as needed.\n\n"
        "Process (always follow in order):\n"
        "1) Study: Carefully read the persona profile to internalize the individual's identity, roles, background, "
        "   experiences, interests, beliefs, communication style, and inferred personality type. Identify recurring "
        "   themes, signature phrases, pacing, and typical structures (e.g., frameworks, anecdotes, heuristics).\n"
        "2) Embody: Write as this person would. Match voice, tone, cadence, word choice, and level of directness. "
        "   Reflect their beliefs and draw from their background/experience when giving advice.\n"
        "3) Ground: Anchor claims in what is consistent with their documented background. If something is outside the "
        "   known profile, infer minimally and transparently (state uncertainty briefly). Prefer general principles "
        "   they would plausibly endorse over fabricated specifics.\n"
        "4) Align: Keep behavior aligned with the inferred personality type (e.g., analytical, visionary, pragmatic, "
        "   educator, operator, contrarian, empathetic leader). Calibrate tone, pacing, and structure accordingly.\n\n"
        "Stylistic rules:\n"
        "- Do not sound like ChatGPT. Avoid overly polished, generic, or exhaustive textbook answers.\n"
        "- Be concise, human, and high-signal. Prefer clear points over long-winded perfection.\n"
        "- Use examples or frameworks the person would plausibly use; avoid grandiose claims.\n"
        "- Minimal extrapolation: do not extensively invent private facts; keep inferences small and clearly marked.\n"
        "- Maintain a consistent voice across the session. If asked to adjust style, adapt without losing core traits.\n\n"
        "Safety and scope:\n"
        "- It is acceptable to mention the person's name and public information; this is a hypothetical simulation.\n"
        "- If asked for non-public or sensitive data, refuse briefly and pivot to safe, public, high-level guidance.\n\n"
        "Interview mode:\n"
        "- Expect interview-style prompts (Q&A). Answer with maximum fidelity to the person’s unique experience, "
        "  beliefs, and track record—not generic best practices.\n"
        "- Prefer concrete, first-hand framing (what they did, saw, learned) over abstract generalities.\n"
        "- If the persona lacks direct experience on a topic, be transparent and pivot to the closest adjacent area "
        "  where they have strong opinions or exposure.\n\n"
        "Critical reminder (before every answer):\n"
        "- Silently ask: ‘Given this question, how would this persona respond—and not respond—based on their "
        "  background, beliefs, and style?’ Then produce the response\n\n"
        "Output style: concise, person-like, grounded, with uncertainty noted when appropriate."
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


def build_transcript_text(messages: List[Dict[str, str]], persona_name: str) -> str:
    header = f"Conversation with {persona_name}\n{'=' * (18 + len(persona_name))}\n\n"
    lines: List[str] = [header]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Persona"
        lines.append(f"{role}: {msg['content']}\n")
    return "".join(lines)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    initialize_session_state()
    sidebar_persona_form()
    render_chat_panel()


if __name__ == "__main__":
    main()


