import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease
from typing import List, Dict
import logging
from models.gpt_model import GPTModel
from models.sentiment_model import SentimentModel
from api.openai_api import OpenAIAPI
import json
import gc

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.openai_api = OpenAIAPI(config.OPENAI_API_KEY)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    async def translate(self, text: str, use_local_models: bool = False) -> Dict[str, str]:
        translations = {}
        try:
            for lang in self.config.TARGET_LANGUAGES:
                if use_local_models:
                    translations[lang] = await self.local_translate(text, lang)
                else:
                    translations[lang] = await self.openai_api.translate(text, lang)
        except Exception as e:
            logger.error(f"Error during translation: {e}")
        finally:
            text = None
            gc.collect()

        return translations

    async def local_translate(self, text: str, target_language: str) -> str:
        # Placeholder for local translation logic
        # Replace this with actual translation code using your local model or service
        logger.debug(f"Translating text to {target_language} using local model.")
        translated_text = f"{text} (translated to {target_language})"
        return translated_text

    async def analyze_sentiment(self, text: str, use_local_models: bool = False) -> Dict[str, float]:
        try:
            if use_local_models:
                sentiment_analysis = SentimentModel().analyze(text)
            else:
                sentiment_analysis = await self.openai_api.analyze_sentiment(text)
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            sentiment_analysis = {}
        finally:
            text = None
            gc.collect()

        return sentiment_analysis

    async def summarize(self, text, model):
        if isinstance(model, GPTModel):
            return await model.summarize(text)
        else:
            raise NotImplementedError("API-based summarization not implemented")

    def preprocess_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
            tokens = [token for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error during text preprocessing: {e}")
            return ""

    def extract_keywords(self, text, top_n=10):
        try:
            preprocessed_text = self.preprocess_text(text)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
            feature_names = vectorizer.get_feature_names_out()
            sorted_items = sorted(zip(tfidf_matrix.tocsc().data, feature_names))
            keywords = [word for _, word in sorted_items[-top_n:]]
            return keywords
        except Exception as e:
            logger.error(f"Error during keyword extraction: {e}")
            return []

    def topic_modeling(self, texts, num_topics=5):
        try:
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
        except Exception as e:
            logger.error(f"Error during topic modeling: {e}")
            return []

    def calculate_readability(self, text):
        try:
            return flesch_reading_ease(text)
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return None

    def count_syllables(self, word):
        return len(
            ''.join(c for c in word if c in 'aeiouAEIOU')
                .replace('es$', '')
                .replace('e$', '')
        )
