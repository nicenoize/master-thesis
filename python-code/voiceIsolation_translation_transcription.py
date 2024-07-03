import os
import argparse
from dotenv import load_dotenv
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
from googletrans import Translator
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class VoiceIsolationTranslator:
    def __init__(self, video_path, target_languages=['de', 'it'], use_cuda=False):
        self.video_path = video_path
        self.target_languages = target_languages
        self.translator = Translator()
        
        # Set device
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use Metal Performance Shaders on M1 Macs
        print(f"Using device: {self.device}")
        
        self.whisper_model = whisper.load_model("base").to(self.device)
        
        # Get the API token from the environment variable
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("Hugging Face token not found in .env file")
        
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                             use_auth_token=hf_token).to(self.device)

    def extract_audio(self):
        print("Extracting audio from video...")
        audio = AudioSegment.from_file(self.video_path)
        audio.export("temp_audio.wav", format="wav")
        return "temp_audio.wav"

    def perform_diarization(self, audio_path):
        print("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        return diarization

    def isolate_voices(self, audio_path, diarization):
        print("Isolating individual voices...")
        y, sr = librosa.load(audio_path)
        isolated_voices = {}

        for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Isolating voices"):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            
            if speaker not in isolated_voices:
                isolated_voices[speaker] = np.zeros_like(y)
            
            isolated_voices[speaker][start_sample:end_sample] = y[start_sample:end_sample]

        for speaker, audio in isolated_voices.items():
            sf.write(f"{speaker}_isolated.wav", audio, sr)

        return isolated_voices

    def transcribe_and_translate(self, isolated_voices):
        print("Transcribing and translating...")
        results = {}
        
        def process_speaker(speaker, audio):
            # Transcribe
            transcription = self.whisper_model.transcribe(audio)
            text = transcription["text"]
            
            # Translate
            translations = {lang: self.translator.translate(text, dest=lang).text
                            for lang in self.target_languages}
            
            return speaker, {
                "transcription": text,
                "translations": translations
            }

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_speaker, speaker, audio) 
                       for speaker, audio in isolated_voices.items()]
            
            for future in tqdm(as_completed(futures), total=len(isolated_voices), desc="Processing speakers"):
                speaker, data = future.result()
                results[speaker] = data

        return results

    def process(self):
        start_time = time.time()
        
        audio_path = self.extract_audio()
        diarization = self.perform_diarization(audio_path)
        isolated_voices = self.isolate_voices(audio_path, diarization)
        results = self.transcribe_and_translate(isolated_voices)

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Isolation and Translation")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--languages", nargs="+", default=['de', 'it'], help="Target languages for translation")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    processor = VoiceIsolationTranslator(args.video_path, target_languages=args.languages, use_cuda=args.use_cuda)
    results = processor.process()

    for speaker, data in results.items():
        print(f"\nSpeaker: {speaker}")
        print(f"Transcription: {data['transcription']}")
        for lang, translation in data['translations'].items():
            print(f"Translation ({lang}): {translation}")