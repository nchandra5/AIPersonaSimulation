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
        "You are researching a real individual based on provided public links and context. Use web search/browsing tools "
        "to gather the most current public information while ensuring identity correctness. Produce a comprehensive, exhaustive "
        "report with as much relevant detail as possible—strive to leave no public detail untouched or omitted from important details to the smallest details. It is "
        "acceptable to mention the person's name and public information. Prefer verifiable sources; when uncertain, mark uncertainty "
        "briefly and avoid fabrication.\n\n"
        "Organize the output with thorough sections (expand each as needed):\n"
        "- Identity: full name, aliases, public roles/titles, affiliations\n"
        "- Background: education, formative experiences, geography, notable milestones (dates where possible)\n"
        "- Experience: roles, orgs, timelines, major projects, outcomes/impact (concise bullets)\n"
        "- Publications & Media: books, papers, blogs, podcasts, interviews, talks (titles, dates)\n"
        "- Interests & Topics: areas of focus, recurring themes, domains of expertise\n"
        "- Communication Style: tone, cadence, preferred formats, signature phrases/structures\n"
        "- Personality & Behavior: likely traits, decision-making patterns, leadership/working style\n"
        "- Representative Quotes/Statements: short quotes or paraphrases with context\n\n"
        "Coverage expectations: be precise, attribute when helpful, include concise timelines, avoid filler. Output your report and no other text."
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

    return profile_text, full_name


