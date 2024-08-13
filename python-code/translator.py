import os
import asyncio
import logging
import io
import time
import gc
import psutil
from dotenv import load_dotenv
from pydub import AudioSegment
import numpy as np
import cv2
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
from pyannote.audio import Pipeline as DiarizationPipeline
from openai import AsyncOpenAI
import rateLimiter

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configurations
CHUNK_DURATION = 3  # Reduced from 5 to 3 seconds
TARGET_LANGUAGES = ['de', 'it']
MAX_MEMORY_USAGE = 16 * 1024 * 1024 * 1024  # 8 GB

class IntegratedVoiceIsolationTranslator:
    def __init__(self, use_local_models=False, use_gpu=True, whisper_model_size="small"):
        self.use_local_models = use_local_models
        self.device = self.get_device(use_gpu)
        logger.info(f"Using device: {self.device}")

        try:
            self.whisper_model, self.whisper_processor = self.load_whisper_model(whisper_model_size)
            self.translation_model, self.translation_tokenizer = self.load_translation_model()
            self.diarization_pipeline = self.load_diarization_pipeline()
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

        if not use_local_models:
            self.aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_device(self, use_gpu):
        if not use_gpu:
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_whisper_model(self, model_size):
        if self.use_local_models:
            model_name = f"openai/whisper-{model_size}"
            model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
            processor = WhisperProcessor.from_pretrained(model_name)
            return model, processor
        else:
            return None, None

    def load_translation_model(self):
        if self.use_local_models:
            model_name = "Helsinki-NLP/opus-mt-en-de"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            return model, tokenizer
        else:
            return None, None

    def load_diarization_pipeline(self):
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("Hugging Face token not found in .env file")
        return DiarizationPipeline.from_pretrained("pyannote/speaker-diarization@2.1", 
                                                   use_auth_token=hf_token).to(self.device)

    async def process_chunk(self, audio_chunk):
        try:
            self.log_memory_usage("Before diarization")
            diarization = await self.perform_diarization(audio_chunk)
            self.log_memory_usage("After diarization")

            isolated_voices = await self.isolate_voices(audio_chunk, diarization)
            self.log_memory_usage("After voice isolation")

            results = await self.transcribe_and_translate(isolated_voices)
            self.log_memory_usage("After transcription and translation")

            self.output_results(results)
            
            # Force garbage collection after processing each chunk
            self.clear_memory()
            
            self.log_memory_usage("After garbage collection")
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")

    async def perform_diarization(self, audio_chunk):
        logger.info("Performing speaker diarization")
        try:
            with io.BytesIO() as audio_file:
                audio_chunk.export(audio_file, format="wav")
                audio_file.seek(0)
                diarization = self.diarization_pipeline(audio_file)
            return diarization
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            raise

    async def isolate_voices(self, audio_chunk, diarization):
        logger.info("Isolating individual voices")
        try:
            audio_array = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
            isolated_voices = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * audio_chunk.frame_rate)
                end_sample = int(turn.end * audio_chunk.frame_rate)
                if speaker not in isolated_voices:
                    isolated_voices[speaker] = []
                isolated_voices[speaker].append(audio_array[start_sample:end_sample])
            return isolated_voices
        except Exception as e:
            logger.error(f"Error isolating voices: {e}")
            raise

    async def transcribe_and_translate(self, isolated_voices):
        logger.info("Transcribing and translating")
        results = {}
        for speaker, audio_chunks in isolated_voices.items():
            try:
                audio = np.concatenate(audio_chunks)
                
                if self.use_local_models:
                    input_features = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                    with torch.no_grad():
                        predicted_ids = self.whisper_model.generate(input_features)
                    transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                else:
                    with io.BytesIO() as audio_file:
                        AudioSegment(audio.tobytes(), frame_rate=16000, sample_width=2, channels=1).export(audio_file, format="mp3")
                        audio_file.seek(0)
                        response = await rateLimiter.api_call_with_backoff_whisper(
                            self.aclient.audio.transcriptions.create,
                            model="whisper-1",
                            file=("audio.mp3", audio_file),
                            response_format="text"
                        )
                        transcription = response
                
                translations = {}
                for lang in TARGET_LANGUAGES:
                    if self.use_local_models:
                        inputs = self.translation_tokenizer(transcription, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            translated_ids = self.translation_model.generate(**inputs)
                        translation = self.translation_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
                    else:
                        response = await rateLimiter.api_call_with_backoff(
                            self.aclient.chat.completions.create,
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"Translate the following text to {lang}. Maintain the speaker labels if present."},
                                {"role": "user", "content": transcription}
                            ],
                            max_tokens=1000
                        )
                        translation = response.choices[0].message.content.strip()
                    translations[lang] = translation
                
                results[speaker] = {
                    "transcription": transcription,
                    "translations": translations
                }
            except Exception as e:
                logger.error(f"Error processing speaker {speaker}: {e}")
        return results

    def extract_audio_chunk(self, video, start_frame, chunk_size):
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(chunk_size):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
            
            audio = np.concatenate([frame[:, :, 0] for frame in frames])
            return AudioSegment(
                audio.tobytes(),
                frame_rate=int(video.get(cv2.CAP_PROP_FPS)),
                sample_width=2,
                channels=1
            )
        except Exception as e:
            logger.error(f"Error extracting audio chunk: {e}")
            raise

    async def run_experiment(self, input_source, experiment_duration=300):
        start_time = time.time()
        try:
            if input_source.startswith('rtmp://'):
                await self.process_stream(input_source)
            else:
                video = cv2.VideoCapture(input_source)
                fps = video.get(cv2.CAP_PROP_FPS)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                chunk_size = int(CHUNK_DURATION * fps)

                for i in range(0, total_frames, chunk_size):
                    if time.time() - start_time > experiment_duration:
                        logger.info("Experiment duration reached. Stopping.")
                        break
                    
                    # Check memory usage and pause if necessary
                    while self.get_memory_usage() > MAX_MEMORY_USAGE:
                        logger.warning("Memory usage too high. Pausing for 5 seconds.")
                        await asyncio.sleep(5)
                        self.clear_memory()
                    
                    audio_chunk = self.extract_audio_chunk(video, i, chunk_size)
                    await self.process_chunk(audio_chunk)

                    self.clear_memory()
                    self.log_memory_usage("After processing chunk")

                video.release()
        except Exception as e:
            logger.error(f"Error in experiment: {e}")
        finally:
            end_time = time.time()
            logger.info(f"Experiment completed. Total time: {end_time - start_time:.2f} seconds")

    def log_memory_usage(self, step=""):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage {step}: RSS: {mem_info.rss / 1024 / 1024:.2f} MB, VMS: {mem_info.vms / 1024 / 1024:.2f} MB")

    def get_memory_usage(self):
        return psutil.virtual_memory().used

    def clear_memory(self):
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()

    def output_results(self, results):
        for speaker, data in results.items():
            logger.info(f"Speaker: {speaker}")
            logger.info(f"Transcription: {data['transcription'][:100]}...")  # Log first 100 chars
            for lang, translation in data['translations'].items():
                logger.info(f"Translation ({lang}): {translation[:100]}...")  # Log first 100 chars