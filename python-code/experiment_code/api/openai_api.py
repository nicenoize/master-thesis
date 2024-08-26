from openai import AsyncOpenAI
import api.rateLimiter as rateLimiter

class OpenAIAPI:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)

    async def transcribe(self, audio_file):
        with open(audio_file, "rb") as audio_file:
            response = await rateLimiter.api_call_with_backoff_whisper(
                self.client.audio.transcriptions.create,
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response

    async def translate(self, text, target_language, model):
        response = await rateLimiter.api_call_with_backoff(
            self.client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_language}. Maintain the speaker labels and format 'Speaker X: [translated text]'."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    async def analyze_sentiment(self, text, model):
        response = await rateLimiter.api_call_with_backoff(
            self.client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": "Perform sentiment analysis on the following text. For each line, respond with a JSON object containing 'speaker', 'sentence', and 'sentiment' (with 'label' and 'score')."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content