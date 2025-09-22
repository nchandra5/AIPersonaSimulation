from typing import Tuple


def build_persona_profile(
    client,
    full_name: str,
    linkedin_url: str,
    x_url: str,
    additional_info: str,
) -> Tuple[str, str]:
    """
    Returns (persona_profile_markdown, redacted_name).

    Uses the Responses API with web-enabled research to synthesize a comprehensive
    persona profile, excluding explicit full-name mentions in the output while
    preserving key background, experience, and behavioral traits.
    """

    research_instructions = (
        "You are researching a real individual based on provided public links and context. "
        "Conduct a broad, careful web-informed synthesis (use browsing/search tools if available) "
        "to produce a persona description that captures background, roles, experience, notable works, "
        "communication style, likely personality traits and behavioral patterns. Do not include the "
        "person's explicit full name anywhere in the output. Summarize neutrally and avoid speculation. "
        "Organize the output with clear sections: Background, Experience, Interests/Topics, Communication Style, "
        "Personality & Behavior, Constraints & Guardrails."
        "Use the browsing/search tools to get the latest information. But ensure that you are gathering information on the correct person."
        "Output your report and no other text."
    )

    user_prompt = (
        f"Inputs Provided:\n"
        f"- Full name: {full_name or 'N/A'}\n"
        f"- LinkedIn: {linkedin_url or 'N/A'}\n"
        f"- X: {x_url or 'N/A'}\n"
        f"- Additional info: {additional_info or 'N/A'}\n\n"
        "Please research and synthesize the persona as specified."
    )

    response = client.responses.create(
        model="gpt-5",  # web-enabled per user's requirement
        reasoning={"effort": "low"},
        tools=[{"type": "web_search"}],
        input=[
            {"role": "developer", "content": research_instructions},
            {"role": "user", "content": user_prompt},
        ],
    )

    profile_text = getattr(response, "output_text", "").strip()

    # Redact the explicit full name in case the model echoed it
    redacted_name = (full_name or "").strip()
    if redacted_name:
        profile_text = profile_text.replace(redacted_name, "[redacted]")
    print(profile_text)
    return profile_text, ("[redacted]" if redacted_name else "")


