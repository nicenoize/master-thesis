import os
import json
import asyncio
import logging
import warnings
from config import get_output_structure
from models.whisper_model import WhisperModel
from models.gpt_model import GPTModel
from models.sentiment_model import SentimentModel
from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.text_processor import TextProcessor
from utils.performance_logger import PerformanceLogger
from utils.plot_generator import generate_plots
import librosa
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip
import gc
import multiprocessing

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")

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
        self.text_processor = TextProcessor(self.config)

    def extract_audio_from_video(self, video_path):
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        video.close()
        return audio_path

    async def run(self):
        whisper_models = self.config.WHISPER_MODELS if self.use_local_models else [self.config.OPENAI_WHISPER_MODEL]
        gpt_models = ["local"] if self.use_local_models else self.config.GPT_MODELS

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

        return self.results

    async def process_video(self, whisper_model, gpt_model):
        output_structure = get_output_structure(
            self.config,
            self.environment, 
            "local" if self.use_local_models else "api",
            whisper_model,
            gpt_model
        )
        
        with self.performance_logger.measure_time(f"total_{whisper_model}_{gpt_model}"):
            video_path = self.input_source
            audio_path = self.extract_audio_from_video(video_path)

            try:
                audio, sampling_rate = sf.read(audio_path)
                audio = audio.astype(np.float32)
            except Exception as sf_error:
                logger.warning(f"Failed to load audio with soundfile: {sf_error}. Trying librosa.")
                try:
                    audio, sampling_rate = librosa.load(audio_path, sr=None)
                except Exception as librosa_error:
                    logger.error(f"Failed to load audio with librosa: {librosa_error}")
                    return None

            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            
            with self.performance_logger.measure_time(f"transcription_{whisper_model}"):
                if self.use_local_models:
                    try:
                        whisper = WhisperModel(whisper_model)
                        transcription = await whisper.transcribe(audio, language='en')
                    except Exception as e:
                        logger.error(f"Error during Whisper model initialization or transcription: {e}")
                        return None
                else:
                    transcription = await self.audio_processor.api_transcribe(audio)
            
            with self.performance_logger.measure_time(f"translation_{gpt_model}"):
                translations = await self.text_processor.translate(transcription, self.use_local_models)
            
            results = {
                "whisper_model": whisper_model,
                "gpt_model": gpt_model,
                "transcription": transcription,
                "translations": translations,
            }

            if self.perform_additional_analysis:
                with self.performance_logger.measure_time(f"sentiment_analysis_{gpt_model}"):
                    results["sentiment_analysis"] = await self.text_processor.analyze_sentiment(transcription, self.use_local_models)
                
                with self.performance_logger.measure_time("video_analysis"):
                    results["video_analysis"] = await self.video_processor.analyze_emotions(self.input_source)
                
                with self.performance_logger.measure_time("audio_analysis"):
                    results["audio_analysis"] = await self.audio_processor.extract_speech_features(audio)
                
                with self.performance_logger.measure_time("text_analysis"):
                    results["text_analysis"] = {
                        "keywords": self.text_processor.extract_keywords(transcription),
                        "readability": self.text_processor.calculate_readability(transcription)
                    }

            # Clear variables after use
            transcription = None
            translations = None
            gc.collect()  # Force garbage collection

            self.save_experiment_results(results, output_structure)
            os.remove(audio_path)

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
