import logging
from openai import AsyncOpenAI
import api.rateLimiter as rateLimiter
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gc


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenAIAPI:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)
        self.api_key = api_key

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
        messages = [
            {"role": "system", "content": f"Translate the following text to {target_language}. Maintain the speaker labels and format 'Speaker X: [translated text]'."},
            {"role": "user", "content": str(text)}  # Ensure text is a string
        ]
        logger.debug(f"Translation request - Model: {model}, Target Language: {target_language}")
        logger.debug(f"Messages: {messages}")
        
        try:
            response = await rateLimiter.api_call_with_backoff(
                self.client.chat.completions.create,
                model=model,
                messages=messages
            )
            logger.debug(f"Translation response: {response}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in translate method: {str(e)}")
            raise

    async def analyze_sentiment(self, text, model):
        messages = [
            {"role": "system", "content": "Perform sentiment analysis on the following text. For each line, respond with a JSON object containing 'speaker', 'sentence', and 'sentiment' (with 'label' and 'score')."},
            {"role": "user", "content": str(text)}  # Ensure text is a string
        ]
        logger.debug(f"Sentiment analysis request - Model: {model}")
        logger.debug(f"Messages: {messages}")
        
        try:
            response = await rateLimiter.api_call_with_backoff(
                self.client.chat.completions.create,
                model=model,
                messages=messages
            )
            logger.debug(f"Sentiment analysis response: {response}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in analyze_sentiment method: {str(e)}")
            raise