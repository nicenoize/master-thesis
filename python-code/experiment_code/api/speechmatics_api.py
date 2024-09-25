# TODO! Implement this
import api.rateLimiter as rateLimiter
import aiohttp
import io

class SpeechmaticsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://asr.api.speechmatics.com/v2"

    async def transcribe(self, audio_bytes, sampling_rate=16000, language='en'):
        # Implement the Speechmatics API call here
        # This is a placeholder implementation
        async with aiohttp.ClientSession() as session:
            # You'll need to implement the actual API call here
            # using the Speechmatics API documentation
            pass
        
        return "Speechmatics transcription result"

    async def translate(self, text, target_language, model):
        # Note: Speechmatics doesn't provide translation, so we're using OpenAI's API here
        openai_api = OpenAIAPI(self.api_key)
        return await openai_api.translate(text, target_language, model)

    async def analyze_sentiment(self, text, model):
        # Note: Speechmatics doesn't provide sentiment analysis, so we're using OpenAI's API here
        openai_api = OpenAIAPI(self.api_key)
        return await openai_api.analyze_sentiment(text, model)

class AudioProcessor:
    def __init__(self, config, api_choice=None):
        self.config = config
        self.api_choice = api_choice
        self.openai_api = OpenAIAPI(config.OPENAI_API_KEY)
        self.speechmatics_api = SpeechmaticsAPI(config.SPEECHMATICS_API_KEY)

    async def api_transcribe(self, audio, sampling_rate=16000, language='en'):
        openai_result = None
        speechmatics_result = None

        # Convert audio to bytes if it's not already
        if isinstance(audio, np.ndarray):
            audio_bytes = io.BytesIO((audio * 32767).astype(np.int16).tobytes())
        elif isinstance(audio, str):
            with open(audio, 'rb') as audio_file:
                audio_bytes = io.BytesIO(audio_file.read())
        elif isinstance(audio, bytes):
            audio_bytes = io.BytesIO(audio)
        else:
            raise ValueError("Unsupported audio format. Expected numpy array, file path, or bytes.")

        if self.api_choice == "1" or self.api_choice == "3":
            openai_result = await self.openai_api.transcribe(audio_bytes, sampling_rate, language)
        
        if self.api_choice == "2" or self.api_choice == "3":
            speechmatics_result = await self.speechmatics_api.transcribe(audio_bytes, sampling_rate, language)

        # Clear the audio data and collect garbage
        audio = None
        audio_bytes = None
        gc.collect()

        return {
            "openai": openai_result,
            "speechmatics": speechmatics_result
        }