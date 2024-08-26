import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease
from typing import List, Dict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

import logging
from typing import List, Dict
from models.gpt_model import GPTModel
from models.sentiment_model import SentimentModel
from api.openai_api import OpenAIAPI
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.openai_api = OpenAIAPI(config.OPENAI_API_KEY)

    async def translate(self, text: str, use_local_models: bool = False) -> Dict[str, str]:
        translations = {}
        for lang in self.config.TARGET_LANGUAGES:
            if use_local_models:
                # Implement local model translation here
                translation = f"[Local Translation to {lang}]: {text}"
            else:
                logger.debug(f"Calling API translation for language: {lang}")
                translation = await self.openai_api.translate(text, lang, self.config.DEFAULT_GPT_MODEL)
            translations[lang] = translation
        return translations

    async def analyze_sentiment(self, text, use_local_models: bool = False):
        if use_local_models:
            # Implement local sentiment analysis here
            return {"positive": 0.5, "negative": 0.3, "neutral": 0.2}
        else:
            logger.debug("Calling API sentiment analysis")
            sentiment_analysis = await self.openai_api.analyze_sentiment(text, self.config.DEFAULT_GPT_MODEL)
            return json.loads(sentiment_analysis)


    async def summarize(self, text, model):
        if isinstance(model, GPTModel):
            return await model.summarize(text)
        else:
            # Implement API-based summarization if needed
            raise NotImplementedError("API-based summarization not implemented")

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def extract_keywords(self, text, top_n=10):
        preprocessed_text = self.preprocess_text(text)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        feature_names = vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(tfidf_matrix.tocsc().data, feature_names))
        keywords = [word for _, word in sorted_items[-top_n:]]
        return keywords

    def topic_modeling(self, texts, num_topics=5):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
        
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        return topics

    def calculate_readability(self, text):
        return flesch_reading_ease(text)

    def count_syllables(self, word):
        return len(
            ''.join(c for c in word if c in 'aeiouAEIOU')
                .replace('es$', '')
                .replace('e$', '')
        )