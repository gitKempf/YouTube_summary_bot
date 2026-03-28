from anthropic import AsyncAnthropic


class SummarizationError(Exception):
    pass


SYSTEM_PROMPT = (
    "You are a skilled narrator writing a voiceover script for a podcast-style summary "
    "of a YouTube video. Write in natural, conversational spoken language — as if a real "
    "person is telling a friend about this video.\n\n"
    "Rules:\n"
    "- NO bullet points, NO headers, NO markdown formatting of any kind\n"
    "- Write flowing paragraphs that sound natural when read aloud\n"
    "- Use transitions like \"Here's the interesting part...\", \"What really stands out is...\", "
    "\"The key thing to take away is...\"\n"
    "- Cover ALL fresh ideas and new insights from the video\n"
    "- Include ALL important information and specific details (numbers, names, examples)\n"
    "- End with a clear takeaway or call to action\n"
    "- Keep the tone engaging and informative, not robotic\n"
    "- Write in the same language as the transcript\n"
    "- Aim for a script that takes 2-3 minutes to read aloud"
)


async def summarize_text(
    text: str,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> str:
    if not text or not text.strip():
        raise ValueError("Cannot summarize empty text")

    client = AsyncAnthropic(api_key=api_key)

    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following transcript:\n\n{text}",
                }
            ],
        )
    except Exception as e:
        raise SummarizationError(f"Anthropic API error: {e}") from e

    return message.content[0].text
