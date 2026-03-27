from anthropic import AsyncAnthropic


class SummarizationError(Exception):
    pass


SYSTEM_PROMPT = (
    "You are an expert content analyst. Summarize the following YouTube video transcript.\n"
    "Your summary must include:\n"
    "1. A one-line overview of what the video is about\n"
    "2. All new ideas and fresh insights presented in the video\n"
    "3. All important information organized by topic\n"
    "4. Key takeaways and actionable points\n\n"
    "Structure the summary with clear headers and bullet points.\n"
    "Keep the summary concise but comprehensive — capture everything valuable.\n"
    "Write in the same language as the transcript."
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
