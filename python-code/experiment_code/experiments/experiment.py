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
import subprocess

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
        del video  # Explicitly delete video object
        gc.collect()  # Force garbage collection
        return audio_path

async def run(self):
    whisper_models = self.config.WHISPER_MODELS if self.use_local_models else [self.config.OPENAI_WHISPER_MODEL]
    gpt_models = ["local"] if self.use_local_models else self.config.GPT_MODELS
    original_bitrate = self.get_audio_bitrate(self.input_source)
    bitrates = [original_bitrate, 8000, 4000]  # Define bitrates lower than the original

    results = []
    for bitrate in bitrates:
        if bitrate <= original_bitrate:
            audio_path = self.convert_audio_bitrate(self.input_source, bitrate)
            for whisper_model in whisper_models:
                for gpt_model in gpt_models:
                    # Process each model and bitrate sequentially
                    result = await self.process_video(whisper_model, gpt_model, audio_path, bitrate)
                    if result is not None:
                        self.results[f"{whisper_model}_{gpt_model}_{bitrate}"] = result
                        results.append(result)
            
            # Clean up the converted audio file to save space
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    self.save_results()
    self.generate_performance_report()
    generate_plots(self.results, self.environment, self.use_local_models)
    self.perform_cross_model_analysis()

    return self.results

def get_audio_bitrate(self, video_path):
    # Use ffprobe or similar tool to get the original bitrate of the audio
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", 
        "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    bitrate = int(result.stdout.decode('utf-8').strip())
    return bitrate

def convert_audio_bitrate(self, video_path, bitrate):
    # Convert the audio from the video to the specified lower bitrate
    audio_path = video_path.rsplit('.', 1)[0] + f'_{bitrate}.wav'
    subprocess.run([
        "ffmpeg", "-i", video_path, "-b:a", str(bitrate), "-ar", "16000", "-ac", "1", audio_path,
        "-y",  # Overwrite without asking
    ], check=True)
    return audio_path

async def process_video(self, whisper_model, gpt_model, audio_path, bitrate):
    output_structure = get_output_structure(
        self.config,
        self.environment, 
        "local" if self.use_local_models else "api",
        whisper_model,
        gpt_model,
        bitrate
    )
    
    with self.performance_logger.measure_time(f"total_{whisper_model}_{gpt_model}_{bitrate}"):
        try:
            # Attempt to read the audio file
            try:
                audio, sampling_rate = sf.read(audio_path)
                audio = audio.astype(np.float32)
            except Exception as sf_error:
                logger.warning(f"SoundFile failed to read the audio: {sf_error}. Trying with librosa.")
                try:
                    audio, sampling_rate = librosa.load(audio_path, sr=None)
                except Exception as librosa_error:
                    logger.error(f"Librosa failed to read the audio: {librosa_error}")
                    return None

            transcription = []
            chunk_size = 16000 * 2  # 2-second chunks at 16kHz
            for i in range(0, len(audio), chunk_size):
                audio_chunk = audio[i:i + chunk_size]
                with self.performance_logger.measure_time(f"transcription_{whisper_model}_{bitrate}"):
                    if self.use_local_models:
                        try:
                            whisper = WhisperModel(whisper_model)
                            transcription_chunk = await whisper.transcribe(audio_chunk, language='en')
                        except Exception as e:
                            logger.error(f"Error during Whisper model transcription: {e}")
                            continue
                    else:
                        transcription_chunk = await self.audio_processor.api_transcribe(audio_chunk)
                    
                    transcription.append(transcription_chunk)

            transcription = " ".join(transcription)

            with self.performance_logger.measure_time(f"translation_{gpt_model}_{bitrate}"):
                translations = await self.text_processor.translate(transcription, self.use_local_models)
            
            results = {
                "whisper_model": whisper_model,
                "gpt_model": gpt_model,
                "bitrate": bitrate,
                "transcription": transcription,
                "translations": translations,
            }

            # Clear variables and force garbage collection
            transcription = None
            translations = None
            audio = None
            gc.collect()

            self.save_experiment_results(results, output_structure)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during audio conversion: {e}")
            return None

        except Exception as e:
            logger.error(f"An error occurred during video processing: {e}")
            return None

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
