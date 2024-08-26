from models.gpt_model import GPTModel
from models.sentiment_model import SentimentModel
from api.openai_api import OpenAIAPI
from config import config
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextProcessor:
    def __init__(self):
        self.openai_api = OpenAIAPI(config["OPENAI_API_KEY"])
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    async def translate(self, text, model):
        if isinstance(model, GPTModel):
            return await model.translate(text, config["TARGET_LANGUAGES"])
        else:
            translations = {}
            for lang in config["TARGET_LANGUAGES"]:
                translations[lang] = await self.openai_api.translate(text, lang, model)
            return translations

    async def analyze_sentiment(self, text, model):
        if isinstance(model, SentimentModel):
            return await model.analyze(text)
        else:
            return await self.openai_api.analyze_sentiment(text, model)

    async def summarize(self, text, model):
        if isinstance(model, GPTModel):
            return await model.summarize(text)
        else:
            return await self.openai_api.summarize(text, model)

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
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self.count_syllables(word) for word in words)
        
        # Calculate Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        
        return fk_grade

    def count_syllables(self, word):
        # This is a simple syllable counter and may not be 100% accurate
        vowels = 'aeiouy'
        num_vowels = 0
        last_was_vowel = False
        for wc in word.lower():
            is_vowel = wc in vowels
            if is_vowel and not last_was_vowel:
                num_vowels += 1
            last_was_vowel = is_vowel
        if word.endswith('e'):
            num_vowels -= 1
        if num_vowels == 0:
            num_vowels = 1
        return num_vowels