import os
import sys
import argparse
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
from googletrans import Translator
import torch
import warnings
from tqdm import tqdm
import time
import gc
import logging
import psutil
import signal

from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class VoiceIsolationTranslator:
    def __init__(self, video_path, target_languages=['de', 'it'], use_gpu=True):
        self.video_path = video_path
        self.target_languages = target_languages
        self.translator = Translator()
        
        self.whisper_device = torch.device("cpu")
        logging.info(f"Using device for Whisper: {self.whisper_device}")
        
        self.device = self.get_device(use_gpu)
        logging.info(f"Using device for other operations: {self.device}")
        
        self.whisper_model = self.load_whisper_model()
        
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("Hugging Face token not found in .env file")
        
        self.diarization_pipeline = self.load_diarization_pipeline(hf_token)

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    def get_device(self, use_gpu):
        if not use_gpu:
            return torch.device("cpu")
        
        if torch.backends.mps.is_available():
            return torch.device("mps")
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        logging.warning("No GPU available. Falling back to CPU.")
        return torch.device("cpu")

    def load_whisper_model(self):
        try:
            model = whisper.load_model("tiny").to(self.whisper_device)
            logging.info(f"Whisper model (tiny) loaded successfully on {self.whisper_device}")
            return model
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise

    def load_diarization_pipeline(self, hf_token):
        try:
            self.log_memory_usage()
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                use_auth_token=hf_token).to(self.device)
            logging.info(f"Diarization pipeline loaded successfully on {self.device}")
            return pipeline
        except Exception as e:
            logging.error(f"Error loading diarization pipeline: {e}")
            raise

    def extract_audio(self):
        self.log_memory_usage()
        logging.info("Extracting audio from video...")
        audio = AudioSegment.from_file(self.video_path)
        audio.export("temp_audio.wav", format="wav")
        return "temp_audio.wav"

    def perform_diarization(self, audio_path):
        self.log_memory_usage()
        logging.info("Performing speaker diarization...")
        try:
            diarization = self.diarization_pipeline(audio_path)
            return diarization
        except Exception as e:
            logging.error(f"Error during diarization: {e}")
            raise

    def isolate_voices(self, audio_path, diarization):
        self.log_memory_usage()
        logging.info("Isolating individual voices...")
        audio_data, sr = sf.read(audio_path)
        isolated_voices = {}

        for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Isolating voices"):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            
            if speaker not in isolated_voices:
                isolated_voices[speaker] = []
            
            isolated_voices[speaker].append(audio_data[start_sample:end_sample])

        return isolated_voices

    def transcribe_and_translate(self, isolated_voices):
        logging.info("Transcribing and translating...")
        results = {}
        
        for speaker, audio_chunks in isolated_voices.items():
            logging.info(f"Processing speaker {speaker}")
            full_text = ""
            
            # Process in smaller chunks
            chunk_size = 3 * 16000  # 3 seconds at 16kHz
            try:
                for i, chunk in enumerate(audio_chunks):
                    logging.info(f"Processing chunk {i+1}/{len(audio_chunks)} for {speaker}")
                    self.log_memory_usage()
                    if len(chunk) > chunk_size:
                        sub_chunks = [chunk[j:j+chunk_size] for j in range(0, len(chunk), chunk_size)]
                    else:
                        sub_chunks = [chunk]
                    
                    for sub_chunk in sub_chunks:
                        try:
                            with time_limit(30):  # 30 seconds timeout
                                transcription = self.whisper_model.transcribe(sub_chunk)
                                full_text += transcription["text"] + " "
                        except TimeoutException:
                            logging.error(f"Transcription timed out for chunk of speaker {speaker}")
                        except Exception as e:
                            logging.error(f"Error transcribing chunk for {speaker}: {e}")
                        
                        # Force garbage collection after each sub-chunk
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        self.log_memory_usage()
                
                translations = {}
                for lang in self.target_languages:
                    try:
                        translations[lang] = self.translator.translate(full_text, dest=lang).text
                    except Exception as e:
                        logging.error(f"Error translating to {lang} for {speaker}: {e}")
                        translations[lang] = "Translation failed"
                
                results[speaker] = {
                    "transcription": full_text.strip(),
                    "translations": translations
                }
            except Exception as e:
                logging.error(f"Error processing speaker {speaker}: {e}")
                results[speaker] = {
                    "transcription": "Transcription failed",
                    "translations": {lang: "Translation failed" for lang in self.target_languages}
                }
            
            # Force garbage collection after each speaker
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.log_memory_usage()

        return results

    def process(self):
        start_time = time.time()
        
        try:
            audio_path = self.extract_audio()
            diarization = self.perform_diarization(audio_path)
            isolated_voices = self.isolate_voices(audio_path, diarization)
            results = self.transcribe_and_translate(isolated_voices)

            end_time = time.time()
            logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

            return results
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            raise
        finally:
            # Clean up temporary files
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            for file in os.listdir():
                if file.endswith("_isolated.wav"):
                    os.remove(file)

def signal_handler(signum, frame):
    logging.error("Received termination signal. Last known state:")
    VoiceIsolationTranslator.log_memory_usage()
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Isolation and Translation")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--languages", nargs="+", default=['de', 'it'], help="Target languages for translation")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        processor = VoiceIsolationTranslator(args.video_path, target_languages=args.languages, use_gpu=args.use_gpu)
        results = processor.process()

        for speaker, data in results.items():
            print(f"\nSpeaker: {speaker}")
            print(f"Transcription: {data['transcription']}")
            for lang, translation in data['translations'].items():
                print(f"Translation ({lang}): {translation}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()