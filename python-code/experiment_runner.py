import os
import argparse
import asyncio
import logging
import time
import json
import csv
from dotenv import load_dotenv
import torch
import numpy as np
import librosa
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
from openai import AsyncOpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import cv2
from fer import FER
import gc
import psutil

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configurations
OUTPUT_DIR = "experiment_results"
CHUNK_DURATION = 5  # seconds
TARGET_LANGUAGES = ['de', 'it']

class ExperimentRunner:
    def __init__(self, video_path, use_gpu=True, perform_additional_analysis=False):
        self.video_path = video_path
        self.use_gpu = use_gpu
        self.perform_additional_analysis = perform_additional_analysis
        self.device = self.get_device()
        self.aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.hf_token = os.getenv('HUGGING_FACE_TOKEN')
        self.diarization_pipeline = self.load_diarization_pipeline()
        self.emotion_detector = FER(mtcnn=True) if perform_additional_analysis else None
        self.results = {}

    def get_device(self):
        if not self.use_gpu:
            return torch.device("cpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load_diarization_pipeline(self):
        return Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=self.hf_token).to(self.device)

    async def run_experiment(self):
        try:
            audio_path = self.extract_audio()
            diarization = self.perform_diarization(audio_path)
            isolated_voices = self.isolate_voices(audio_path, diarization)

            gpt_models = ["gpt-3.5-turbo", "gpt-4"]
            whisper_models = ["tiny", "base", "small", "medium"]

            for gpt_model in gpt_models:
                for whisper_model in whisper_models:
                    logger.info(f"Running experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}")
                    start_time = time.time()
                    
                    try:
                        transcriptions = await self.transcribe(isolated_voices, whisper_model)
                        translations = await self.translate(transcriptions, gpt_model)
                        
                        if self.perform_additional_analysis:
                            sentiment_analysis = await self.analyze_sentiment(transcriptions)
                            video_analysis = self.analyze_video()
                        else:
                            sentiment_analysis = None
                            video_analysis = None

                        end_time = time.time()
                        total_time = end_time - start_time

                        self.results[f"{gpt_model}_{whisper_model}"] = {
                            "transcriptions": transcriptions,
                            "translations": translations,
                            "sentiment_analysis": sentiment_analysis,
                            "video_analysis": video_analysis,
                            "processing_time": total_time
                        }

                        self.save_results(gpt_model, whisper_model)
                    except Exception as e:
                        logger.error(f"Error processing {gpt_model}_{whisper_model}: {str(e)}")

            self.generate_performance_report()
        except Exception as e:
            logger.error(f"Error in run_experiment: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def extract_audio(self):
        logger.info("Extracting audio from video...")
        audio = AudioSegment.from_file(self.video_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output directory if it doesn't exist
        audio_path = os.path.join(OUTPUT_DIR, "temp_audio.wav")
        audio.export(audio_path, format="wav")
        return audio_path

    def perform_diarization(self, audio_path):
        logger.info("Performing speaker diarization...")
        return self.diarization_pipeline(audio_path)

    def isolate_voices(self, audio_path, diarization):
        logger.info("Isolating individual voices...")
        audio_data, sr = librosa.load(audio_path)  # Use librosa instead of sf
        isolated_voices = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            
            if speaker not in isolated_voices:
                isolated_voices[speaker] = []
            
            isolated_voices[speaker].append(audio_data[start_sample:end_sample])

        return isolated_voices

    async def transcribe(self, isolated_voices, model_name):
        logger.info(f"Transcribing with Whisper model: {model_name}")
        model = whisper.load_model(model_name).to(self.device)
        transcriptions = {}

        for speaker, audio_chunks in isolated_voices.items():
            full_text = ""
            for chunk in audio_chunks:
                result = model.transcribe(chunk)
                full_text += result["text"] + " "
            transcriptions[speaker] = full_text.strip()

        return transcriptions

    async def translate(self, transcriptions, model_name):
        logger.info(f"Translating with GPT model: {model_name}")
        translations = {lang: {} for lang in TARGET_LANGUAGES}

        for speaker, text in transcriptions.items():
            for lang in TARGET_LANGUAGES:
                response = await self.aclient.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": f"Translate the following text to {lang}. Maintain the speaker labels if present."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=1000
                )
                translations[lang][speaker] = response.choices[0].message.content.strip()

        return translations

    async def analyze_sentiment(self, transcriptions):
        logger.info("Performing sentiment analysis...")
        sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
        sentiments = {}

        for speaker, text in transcriptions.items():
            sentiment = sentiment_analyzer(text)[0]
            sentiments[speaker] = {
                "label": sentiment["label"],
                "score": sentiment["score"]
            }

        return sentiments

    def analyze_video(self):
        logger.info("Analyzing video frames...")
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        emotions = []
        for i in range(0, frame_count, int(fps)):  # Analyze one frame per second
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                emotion = self.emotion_detector.detect_emotions(frame)
                if emotion:
                    emotions.append(emotion[0]['emotions'])

        video.release()
        return {
            "duration": duration,
            "emotions": emotions
        }

    def save_results(self, gpt_model, whisper_model):
        model_dir = os.path.join(OUTPUT_DIR, f"{gpt_model}_{whisper_model}")
        os.makedirs(model_dir, exist_ok=True)

        # Save transcriptions
        with open(os.path.join(model_dir, "transcriptions.json"), "w") as f:
            json.dump(self.results[f"{gpt_model}_{whisper_model}"]["transcriptions"], f, indent=2)

        # Save translations
        for lang in TARGET_LANGUAGES:
            with open(os.path.join(model_dir, f"translations_{lang}.json"), "w") as f:
                json.dump(self.results[f"{gpt_model}_{whisper_model}"]["translations"][lang], f, indent=2)

        # Save additional analysis if performed
        if self.perform_additional_analysis:
            with open(os.path.join(model_dir, "sentiment_analysis.json"), "w") as f:
                json.dump(self.results[f"{gpt_model}_{whisper_model}"]["sentiment_analysis"], f, indent=2)
            with open(os.path.join(model_dir, "video_analysis.json"), "w") as f:
                json.dump(self.results[f"{gpt_model}_{whisper_model}"]["video_analysis"], f, indent=2)

    def generate_performance_report(self):
        report_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
        with open(report_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["GPT Model", "Whisper Model", "Processing Time (s)"])
            for key, value in self.results.items():
                gpt_model, whisper_model = key.split("_")
                writer.writerow([gpt_model, whisper_model, value["processing_time"]])

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

async def main():
    parser = argparse.ArgumentParser(description="Transcription and Translation Experiment Runner")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--perform_additional_analysis", action="store_true", help="Perform additional sentiment and video analysis")
    args = parser.parse_args()

    runner = ExperimentRunner(args.video_path, args.use_gpu, args.perform_additional_analysis)
    await runner.run_experiment()

if __name__ == "__main__":
    asyncio.run(main())