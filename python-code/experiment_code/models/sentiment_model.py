from transformers import pipeline

class SentimentModel:
    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    async def analyze(self, text):
        result = self.model(text)
        return result[0]