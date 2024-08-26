import os
import json
import asyncio
from config import get_output_structure
from models.whisper_model import WhisperModel
from models.gpt_model import GPTModel
from models.sentiment_model import SentimentModel
from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.text_processor import TextProcessor
from utils.performance_logger import PerformanceLogger
from utils.plot_generator import generate_plots

class Experiment:
    def __init__(self, config, input_source, use_local_models, model_choice, perform_additional_analysis, environment, performance_logger, api_choice=None):
        self.config = config
        self.input_source = input_source
        self.use_local_models = use_local_models
        self.model_choice = model_choice
        self.perform_additional_analysis = perform_additional_analysis
        self.environment = environment
        self.performance_logger = performance_logger
        self.api_choice = api_choice
        self.results = {}
        
        self.audio_processor = AudioProcessor(self.config, self.api_choice)
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor()

    async def run(self):
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        gpt_models = ["local"] if self.use_local_models else ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

        tasks = []
        for whisper_model in whisper_models:
            for gpt_model in gpt_models:
                tasks.append(self.process_video(whisper_model, gpt_model))

        results = await asyncio.gather(*tasks)
        for result in results:
            self.results[f"{result['whisper_model']}_{result['gpt_model']}"] = result

        self.save_results()
        self.generate_performance_report()
        generate_plots(self.results, self.environment, self.use_local_models)
        self.perform_cross_model_analysis()

    async def process_video(self, whisper_model, gpt_model):
        output_structure = get_output_structure(
            self.environment, 
            "local" if self.use_local_models else "api",
            whisper_model,
            gpt_model
        )
        
        with self.performance_logger.measure_time(f"total_{whisper_model}_{gpt_model}"):
            audio = self.audio_processor.extract_audio(self.input_source)
            
            whisper = WhisperModel(whisper_model) if self.use_local_models else None
            gpt = GPTModel(gpt_model) if self.use_local_models else None
            sentiment = SentimentModel() if self.use_local_models else None

            with self.performance_logger.measure_time(f"transcription_{whisper_model}"):
                transcription = await whisper.transcribe(audio) if self.use_local_models else await self.audio_processor.api_transcribe(audio)
            
            with self.performance_logger.measure_time(f"translation_{gpt_model}"):
                translations = await self.text_processor.translate(transcription, gpt)
            
            results = {
                "whisper_model": whisper_model,
                "gpt_model": gpt_model,
                "transcription": transcription,
                "translations": translations,
            }

            if self.perform_additional_analysis:
                with self.performance_logger.measure_time(f"sentiment_analysis_{gpt_model}"):
                    results["sentiment_analysis"] = await self.text_processor.analyze_sentiment(transcription, sentiment)
                
                with self.performance_logger.measure_time("video_analysis"):
                    results["video_analysis"] = await self.video_processor.analyze_emotions(self.input_source)
                
                with self.performance_logger.measure_time("audio_analysis"):
                    results["audio_analysis"] = await self.audio_processor.extract_speech_features(audio)
                
                with self.performance_logger.measure_time("text_analysis"):
                    results["text_analysis"] = {
                        "keywords": self.text_processor.extract_keywords(transcription),
                        "readability": self.text_processor.calculate_readability(transcription)
                    }

            self.save_experiment_results(results, output_structure)
        return results

    def save_experiment_results(self, results, output_structure):
        for key, path in output_structure.items():
            os.makedirs(path, exist_ok=True)
            if key in results:
                with open(os.path.join(path, f"{key}.json"), "w") as f:
                    json.dump(results[key], f, indent=2)

    def save_results(self):
        with open(f"results_{self.environment}.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def generate_performance_report(self):
        self.performance_logger.generate_report(self.environment, self.use_local_models)

    def perform_cross_model_analysis(self):
        # Analyze transcription consistency across Whisper models
        transcriptions = [result['transcription'] for result in self.results.values()]
        consistency_scores = self.text_processor.calculate_consistency(transcriptions)
        
        # Compare sentiment analysis results across models
        sentiment_results = [result.get('sentiment_analysis') for result in self.results.values() if 'sentiment_analysis' in result]
        sentiment_agreement = self.text_processor.calculate_sentiment_agreement(sentiment_results)
        
        # Analyze translation quality (if reference translations are available)
        translation_scores = self.text_processor.evaluate_translations(self.results)
        
        cross_model_analysis = {
            "transcription_consistency": consistency_scores,
            "sentiment_agreement": sentiment_agreement,
            "translation_quality": translation_scores
        }
        
        with open(f"cross_model_analysis_{self.environment}.json", "w") as f:
            json.dump(cross_model_analysis, f, indent=2)