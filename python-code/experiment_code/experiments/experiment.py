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
import subprocess
import psutil  # For system resource optimization
import pandas as pd
import asyncio
from api.rateLimiter import gpt_rate_limiter, whisper_rate_limiter, api_call_with_backoff_whisper

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")

class Experiment:
    def __init__(self, config, input_source, use_local_models, model_choice, perform_additional_analysis, environment, performance_logger, api_choice=None, use_diarization=True):
        self.config = config
        self.input_source = input_source
        self.use_local_models = use_local_models
        self.model_choice = model_choice
        self.perform_additional_analysis = perform_additional_analysis
        self.environment = environment
        self.performance_logger = performance_logger
        self.api_choice = api_choice
        self.use_diarization = use_diarization
        self.results = {}
        
        self.audio_processor = AudioProcessor(self.config, self.api_choice)
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor(self.config)

        self.available_cores = psutil.cpu_count(logical=False)  # Get available physical cores

    async def extract_audio_from_video(video_path, bitrate):
        audio_output_path = video_path.replace(".mp4", f"_audio_{bitrate}.wav")

        if not os.path.exists(audio_output_path):
            logging.info(f"Extracting audio from video: {video_path}")
            command = [
                "ffmpeg", "-i", video_path, "-vn",  # "-vn" ensures only audio is extracted
                "-acodec", "pcm_s16le",  # Encode to PCM (uncompressed audio)
                "-ar", "16000",  # Sampling rate of 16000 Hz
                "-ac", "1",  # Mono channel
                audio_output_path
            ]
            try:
                subprocess.run(command, check=True)
                logging.info(f"Audio extracted to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio: {e}")
                return None
        else:
            logging.info(f"Audio file {audio_output_path} already exists. Skipping extraction.")
        
        return audio_output_path

    async def run(self):
        whisper_models = self.config.WHISPER_MODELS if self.use_local_models else [self.config.OPENAI_WHISPER_MODEL]
        gpt_models = ["local"] if self.use_local_models else self.config.GPT_MODELS
        original_bitrate = self.get_audio_bitrate(self.input_source)
        bitrates = [original_bitrate, 8000, 4000]

        results = []
        for bitrate in bitrates:
            if bitrate <= original_bitrate:
                audio_path = self.convert_audio_bitrate(self.input_source, bitrate)
                logger.info(f"Processing audio at bitrate: {bitrate}")
                for whisper_model in whisper_models:
                    for gpt_model in gpt_models:
                        logger.info(f"Processing with Whisper model: {whisper_model}, GPT model: {gpt_model}")
                        result = await self.process_video(whisper_model, gpt_model, audio_path, bitrate)
                        if result is not None:
                            result_entry = {
                                'Whisper Model': whisper_model,
                                'GPT Model': gpt_model,
                                'Bitrate': bitrate,
                                'transcription': result['transcription'],
                                'translations': result['translations'],
                                'sentiment_analysis': result.get('sentiment_analysis', []),
                                'text_analysis': result.get('text_analysis', {}),
                                'performance_logs': {
                                    'transcription': self.performance_logger.logs.get(f"transcription_{whisper_model}_{bitrate}", [])[-1],
                                    'translation': self.performance_logger.logs.get(f"translation_{gpt_model}_{bitrate}", [])[-1],
                                    'total': self.performance_logger.logs.get(f"total_{whisper_model}_{gpt_model}_{bitrate}", [])[-1]
                                }
                            }
                            results.append(result_entry)
                            self.results[f"{whisper_model}_{gpt_model}_{bitrate}"] = result_entry
                
                # Clean up the converted audio file to save space
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        
        results_df = pd.DataFrame(results)
        
        self.save_results()
        self.generate_performance_report()
        generate_plots(results_df, self.environment, self.use_local_models, self.performance_logger.logs)
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

    def convert_audio_bitrate(input_path, output_path, bitrate=16000):

        # Converts the audio to the specified bitrate using ffmpeg.
        logging.info(f"Converting audio file {input_path} to {bitrate} Hz")
        
        if os.path.exists(output_path):
            logging.info(f"Output file {output_path} already exists. Skipping conversion.")
            return output_path
        
        # Use ffmpeg to convert the audio bitrate and channels
        command = [
            'ffmpeg', '-i', input_path,
            '-ar', str(bitrate),  # set the audio sampling rate
            '-ac', '1',  # set the audio channels to mono
            output_path
        ]
        
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"Conversion result: {result.stdout.decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting audio: {e.stderr.decode('utf-8')}")
            return None
        
        return output_path

    async def process_video(self, whisper_model, gpt_model, video_path, bitrate):
        """
        Processes a video file to extract, transcribe, and analyze the audio, with rate limiting for API requests.
        """
        audio_output_path = video_path.replace(".mp4", f"_{bitrate}.wav")

        if not os.path.exists(audio_output_path):
            logging.info(f"Extracting audio from video: {video_path}")
            command = [
                "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1",
                "-f", "wav", audio_output_path
            ]
            try:
                subprocess.run(command, check=True)
                logging.info(f"Audio extracted to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error extracting audio: {e}")
                return None
        else:
            logging.info(f"Audio file {audio_output_path} already exists. Skipping extraction.")

        with self.performance_logger.measure_time(f"total_{whisper_model}_{gpt_model}_{bitrate}"):
            try:
                try:
                    audio, sampling_rate = sf.read(audio_output_path)
                    audio = audio.astype(np.float32)
                    logging.info(f"Loaded audio with shape: {audio.shape} and sampling rate: {sampling_rate}")
                except Exception as sf_error:
                    logger.warning(f"SoundFile failed: {sf_error}. Trying with librosa.")
                    try:
                        audio, sampling_rate = librosa.load(audio_output_path, sr=None)
                        logging.info(f"Loaded audio using librosa with shape: {audio.shape}")
                    except Exception as librosa_error:
                        logger.error(f"Librosa failed to read the audio: {librosa_error}")
                        return None

                if sampling_rate != 16000 or len(audio.shape) != 1:
                    logging.error("Audio file format incorrect.")
                    return None

                chunk_duration = 30  # seconds per chunk
                chunk_size = int(chunk_duration * sampling_rate)
                logging.info(f"Processing audio in chunks of size: {chunk_size} (approx {chunk_duration} seconds per chunk)")

                # Use an asyncio semaphore to limit parallelism
                semaphore = asyncio.Semaphore(10)  # limit to 10 parallel API calls

                transcription_tasks = []

                for i in range(0, len(audio), chunk_size):
                    audio_chunk = audio[i:i + chunk_size]
                    logging.info(f"Processing chunk {i // chunk_size + 1}/{len(audio) // chunk_size + 1}")

                    if not np.any(audio_chunk):
                        logging.warning(f"Chunk {i // chunk_size + 1} contains only zeros.")
                        continue

                    transcription_task = asyncio.ensure_future(
                        self.transcribe_chunk_with_rate_limiting(
                            whisper_model, gpt_model, audio_chunk, sampling_rate, semaphore
                        )
                    )
                    transcription_tasks.append(transcription_task)

                transcriptions = await asyncio.gather(*transcription_tasks)
                transcription = " ".join([t for t in transcriptions if t])

                # Run translation and other processing if necessary
                with self.performance_logger.measure_time(f"translation_{gpt_model}_{bitrate}"):
                    translations = await self.text_processor.translate(transcription, self.use_local_models)

                results = {
                    "whisper_model": whisper_model,
                    "gpt_model": gpt_model,
                    "bitrate": bitrate,
                    "transcription": transcription,
                    "translations": translations,
                }

                if self.perform_additional_analysis:
                    with self.performance_logger.measure_time(f"sentiment_analysis_{whisper_model}_{gpt_model}_{bitrate}"):
                        sentiment_model = SentimentModel()
                        results["sentiment_analysis"] = sentiment_model.analyze(transcription)

                    with self.performance_logger.measure_time(f"text_analysis_{whisper_model}_{gpt_model}_{bitrate}"):
                        results["text_analysis"] = self.text_processor.analyze_text(transcription)

                audio = None
                gc.collect()

                self.save_experiment_results(results, output_structure)

            except subprocess.CalledProcessError as e:
                logger.error(f"Error during audio conversion: {e}")
                return None

            except Exception as e:
                logger.error(f"An error occurred: {e}")
                return None

        return results


    async def transcribe_chunk_with_rate_limiting(self, whisper_model, gpt_model, audio_chunk, sampling_rate, semaphore):
        """
        Transcribes a chunk of audio with rate limiting and concurrency control.
        """
        async with semaphore:  # Ensures a limit on concurrent API calls
            try:
                await whisper_rate_limiter.acquire()
                with self.performance_logger.measure_time(f"transcription_{whisper_model}"):
                    transcription_chunk = await api_call_with_backoff_whisper(
                        self.audio_processor.api_transcribe,
                        audio_chunk, sampling_rate
                    )
                return transcription_chunk
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                return ""


    def get_dynamic_chunk_size(self, total_length):
        available_memory = psutil.virtual_memory().available
        # Estimate chunk size based on available memory and total length of the audio
        chunk_size = min(total_length, int(available_memory // (16 * 1024 * 1024)))  # Rough estimate, 16MB per chunk
        return chunk_size

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
