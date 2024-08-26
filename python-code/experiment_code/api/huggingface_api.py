from transformers import pipeline
import torch

class HuggingFaceAPI:
    def __init__(self, token):
        self.token = token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def transcribe(self, audio_file):
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=self.device)
        result = transcriber(audio_file)
        return result["text"]

    async def translate(self, text, target_language):
        translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_language}", device=self.device)
        result = translator(text)
        return result[0]["translation_text"]

    async def analyze_sentiment(self, text):
        sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
        result = sentiment_analyzer(text)
        return result[0]