import logging

logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self):
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline("sentiment-analysis")
        except ImportError as e:
            logger.error(f"Error importing transformers pipeline: {e}")
            self.sentiment_pipeline = None

    async def analyze(self, text):
        if self.sentiment_pipeline is None:
            logger.error("Sentiment analysis pipeline is not available.")
            return None
        
        try:
            result = self.sentiment_pipeline(text)
            return result[0]
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            return None