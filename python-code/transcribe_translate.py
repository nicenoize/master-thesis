import os
import asyncio
import logging
import io
import time
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import detect_silence
import cv2
import csv
import numpy as np
from fer import FER
import librosa
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
import tiktoken
import tenacity
import subprocess
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import torch
from scipy import stats
import signal
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import rateLimiter
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import json
import re
import soundfile as sf
from moviepy.editor import VideoFileClip



os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Initialize FER
emotion_detector = FER(mtcnn=True)

# Global variables to track progress
current_gpt_model = None
current_whisper_model = None
experiment_completed = False

def initialize_transcription_pipeline(model_name="openai/whisper-large"):
    return pipeline("automatic-speech-recognition", model=model_name, device=device)

def load_audio(file_path, sr=16000):
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono
    return y, sr

async def extract_speech_features(audio_chunk):

    if len(audio_chunk) < 100:  # Skip chunks shorter than 100 ms
        logger.warning(f"Skipping speech feature extraction for chunk with duration {len(audio_chunk)} ms (too short)")
        return None
    # Convert pydub AudioSegment to numpy array
    audio_array = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_chunk.frame_rate

    # Detect pauses
    silence_thresh = -30  # dB
    min_silence_len = 100  # ms
    silences = detect_silence(audio_chunk, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    pauses = [{"start": start / 1000, "end": end / 1000} for start, end in silences]

    # Extract pitch and intonation
    sound = parselmouth.Sound(audio_array, sampling_frequency=sample_rate)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch.selected_array['frequency']
    pitch_mean = np.mean(pitch_values[pitch_values != 0])
    pitch_std = np.std(pitch_values[pitch_values != 0])

    # Extract intensity (volume)
    intensity = sound.to_intensity()
    intensity_values = intensity.values[0]
    intensity_mean = np.mean(intensity_values)
    intensity_std = np.std(intensity_values)

    # Estimate speech rate using zero-crossings
    zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
    duration = len(audio_array) / sample_rate
    speech_rate = zero_crossings / (2 * duration)  # Rough estimate of syllables per second

    # Extract formants for vowel analysis
    formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1_mean = call(formants, "Get mean", 1, 0, 0, "hertz")
    f2_mean = call(formants, "Get mean", 2, 0, 0, "hertz")

    return {
        "pauses": pauses,
        "pitch": {"mean": pitch_mean, "std": pitch_std},
        "intensity": {"mean": intensity_mean, "std": intensity_std},
        "speech_rate": speech_rate,
        "formants": {"F1": f1_mean, "F2": f2_mean}
    }


async def analyze_speech_characteristics(audio_features):
    pitch = audio_features["pitch"]
    intensity = audio_features["intensity"]
    speech_rate = audio_features["speech_rate"]
    
    analysis = []
    
    # Analyze pitch
    if pitch["mean"] > 150:
        analysis.append("The speaker's voice is relatively high-pitched.")
    elif pitch["mean"] < 100:
        analysis.append("The speaker's voice is relatively low-pitched.")
    
    if pitch["std"] > 30:
        analysis.append("There's significant pitch variation, indicating an expressive or emotional speaking style.")
    elif pitch["std"] < 10:
        analysis.append("The pitch is relatively monotone, suggesting a calm or reserved speaking style.")
    
    # Analyze intensity
    if intensity["std"] > 10:
        analysis.append("The speaker uses notable volume changes, possibly for emphasis.")
    elif intensity["std"] < 5:
        analysis.append("The speaker maintains a consistent volume throughout.")
    
    # Analyze speech rate
    if speech_rate > 4:
        analysis.append("The speaker is talking quite rapidly.")
    elif speech_rate < 2:
        analysis.append("The speaker is talking slowly and deliberately.")
    
    # Analyze pauses
    if len(audio_features["pauses"]) > 5:
        analysis.append("The speech contains frequent pauses, possibly for emphasis or thoughtful consideration.")
    elif len(audio_features["pauses"]) < 2:
        analysis.append("The speech flows continuously with few pauses.")
    
    return " ".join(analysis)


def validate_file_path(file_path):
    if not file_path:
        return False, "File path is empty."
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    return True, ""

# Signal handler function
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Saving current state and exiting...")
    save_current_state()
    sys.exit(0)

import json
import logging

async def analyze_sentiment_per_sentence(text, use_local_model=False):
    logger.info("Performing sentiment analysis per sentence")
    if text is None:
        logger.error("Input text for sentiment analysis is None")
        return None
    
    lines = text.split('\n')
    results = []
    try:
        for line in lines:
            parts = line.split(': ', 1)
            if len(parts) == 2:
                speaker, sentence = parts
            else:
                speaker, sentence = "Unknown", line

            if use_local_model:
                # Use local sentiment analysis model
                sentiment = sentiment_analyzer(sentence)[0]
                results.append({
                    "speaker": speaker,
                    "sentence": sentence,
                    "sentiment": {"label": sentiment["label"], "score": sentiment["score"]}
                })
            else:
                # Use OpenAI API for sentiment analysis
                try:
                    response = await rateLimiter.api_call_with_backoff(
                        aclient.chat.completions.create,
                        model=current_gpt_model,
                        messages=[
                            {"role": "system", "content": "Perform sentiment analysis on the following sentence. Respond with a JSON object containing 'label' (positive, negative, or neutral) and 'score' (between 0 and 1)."},
                            {"role": "user", "content": sentence}
                        ]
                    )
                    response_content = response.choices[0].message.content.strip()
                    
                    # Attempt to parse JSON, if fails, extract information manually
                    try:
                        sentiment = json.loads(response_content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract information manually
                        label = "unknown"
                        score = 0.5
                        if "positive" in response_content.lower():
                            label = "positive"
                        elif "negative" in response_content.lower():
                            label = "negative"
                        elif "neutral" in response_content.lower():
                            label = "neutral"
                        
                        # Try to extract score
                        score_match = re.search(r'score"?\s*:\s*(\d+(\.\d+)?)', response_content)
                        if score_match:
                            score = float(score_match.group(1))
                        
                        sentiment = {"label": label, "score": score}
                    
                    results.append({
                        "speaker": speaker,
                        "sentence": sentence,
                        "sentiment": sentiment
                    })
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for sentence: {sentence}. Error: {str(e)}")
                    results.append({
                        "speaker": speaker,
                        "sentence": sentence,
                        "sentiment": {"label": "error", "score": 0.5}
                    })
        return results
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return None

# Function to save current state
def save_current_state():
    global experiment_completed
    conversation.save_to_files()
    save_performance_logs()
    generate_performance_plots()
    if not experiment_completed:
        with open(os.path.join(OUTPUT_DIR, "incomplete_experiment.txt"), "w") as f:
            f.write(f"Experiment interrupted.\nLast models used: GPT - {current_gpt_model}, Whisper - {current_whisper_model}")


if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

CHUNK_SIZE = 16000 * 10 * 2  # 5 seconds of audio at 16kHz, 16-bit
TARGET_LANGUAGES = ['ger']  # Only German for testing
OUTPUT_DIR = "output"
MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB, just under OpenAI's 26 MB limit

# Rate limiting
rate_limit = AsyncLimiter(10, 60)  # 10 requests per minute

# Queue for chunk processing
chunk_queue = asyncio.Queue()

# Performance logging
performance_logs = {
    "transcription": {},
    "translation": {},
    "analysis": {},
    "total_processing": {}
}

# Environment and model tracking
current_environment = "M1 Max"
current_gpt_model = "gpt-4"
current_whisper_model = "large"

class Conversation:
    def __init__(self):
        self.transcriptions = {}
        self.translations = {lang: {} for lang in TARGET_LANGUAGES}

    def add_transcription(self, model_key, text):
        self.transcriptions.setdefault(model_key, []).append(text)

    def add_translation(self, model_key, lang, text):
        self.translations[lang].setdefault(model_key, []).append(text)

    def save_to_files(self):
        base_dir = os.path.join(OUTPUT_DIR, "transcriptions_and_translations")
        os.makedirs(base_dir, exist_ok=True)

        for model_key, texts in self.transcriptions.items():
            file_path = os.path.join(base_dir, f"transcription_{model_key}.txt")
            with open(file_path, "w") as f:
                f.write("\n\n".join(texts))

        for lang in TARGET_LANGUAGES:
            lang_dir = os.path.join(base_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            for model_key, texts in self.translations[lang].items():
                file_path = os.path.join(lang_dir, f"translation_{model_key}.txt")
                with open(file_path, "w") as f:
                    f.write("\n\n".join(texts))

conversation = Conversation()

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(10),
    retry=tenacity.retry_if_exception_type((Exception, tenacity.TryAgain))
)

async def analyze_audio_features(audio_chunk):
    audio_array = np.array(audio_chunk.get_array_of_samples())
    mfccs = librosa.feature.mfcc(y=audio_array.astype(float), sr=audio_chunk.frame_rate)
    chroma = librosa.feature.chroma_stft(y=audio_array.astype(float), sr=audio_chunk.frame_rate)
    return {
        "mfccs": np.mean(mfccs, axis=1).tolist(),
        "chroma": np.mean(chroma, axis=1).tolist()
    }

async def analyze_video_frame(frame):
    emotions = emotion_detector.detect_emotions(frame)
    return emotions[0]['emotions'] if emotions else None

sentiment_analyzer = pipeline("sentiment-analysis", device="cuda" if torch.cuda.is_available() else "cpu")

async def detailed_analysis(transcription, audio_features, speech_features, speech_analysis, video_emotions, use_local_models=False):
    logger.info("Performing detailed analysis.")
    start_time = time.time()
    try:
        sentiment_results = await analyze_sentiment_per_sentence(transcription, use_local_models)
        
        if use_local_models:
            # Perform local sentiment analysis
            analysis_result = f"Transcription:\n{transcription}\n\n"
            analysis_result += f"Sentiment Analysis:\n"
            for result in sentiment_results:
                analysis_result += f"  {result['speaker']}: {result['sentence']}\n"
                analysis_result += f"    Sentiment: {result['sentiment']['label']} (score: {result['sentiment']['score']:.2f})\n"
            analysis_result += f"\nAudio Features: {audio_features}\n"
            analysis_result += f"Speech Features: {speech_features}\n"
            analysis_result += f"Speech Analysis: {speech_analysis}\n"
            analysis_result += f"Video Emotions: {video_emotions}\n"
        else:
            # Use OpenAI API
            analysis_prompt = f"""
            Analyze the following transcription, taking into account the provided audio features, speech characteristics, video emotions, and per-sentence sentiment analysis:

            Transcription:
            {transcription}

            Sentiment Analysis:
            {json.dumps(sentiment_results, indent=2)}

            Audio Features: {audio_features}

            Speech Features: {speech_features}

            Speech Analysis: {speech_analysis}

            Video Emotions: {video_emotions}

            Based on this information:
            1. For each speaker, provide a summary of their speaking style, including intonation, emphasis, and emotional trends.
            2. Note any significant emotional changes or discrepancies between speech content and audio/visual cues.
            3. Provide an overall analysis of the conversation, including the dynamics between speakers.

            Format your response as:
            Speaker X Summary:
            [Summary of speaking style and emotions]

            Overall Analysis:
            [Overall analysis of the conversation]
            """

            try:
                response = await rateLimiter.api_call_with_backoff(
                    aclient.chat.completions.create,
                    model=current_gpt_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in multimodal sentiment analysis, capable of interpreting text, audio features, and visual emotional cues."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=2000
                )
                analysis_result = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error during detailed analysis API call: {e}")
                analysis_result = f"Error during detailed analysis. Raw sentiment results: {json.dumps(sentiment_results, indent=2)}"

        performance_logs["analysis"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return analysis_result
    except Exception as e:
        logger.error(f"Error during detailed analysis: {e}", exc_info=True)
        performance_logs["analysis"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return f"Error during analysis. Transcription: {transcription}"


async def transcribe_audio(input_source, whisper_model, use_local_model=False):
    logger.info(f"Starting transcription with Whisper model: {whisper_model}. Use local model: {use_local_model}")
    start_time = time.time()
    try:
        # Extract audio from video
        video = VideoFileClip(input_source)
        audio = video.audio
        audio_file = input_source.rsplit('.', 1)[0] + '.wav'
        audio.write_audiofile(audio_file, codec='pcm_s16le')
        video.close()

        logger.info(f"Audio extracted from video: {audio_file}")

        # Resample audio to 16000 Hz
        audio, sr = librosa.load(audio_file, sr=16000)
        sf.write(audio_file, audio, sr, subtype='PCM_16')

        logger.info(f"Audio resampled to 16000 Hz")

        if use_local_model:
            # Use local Whisper model
            device = torch.device("cpu")
            model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_model}").to(device)
            processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model}")
            
            logger.info(f"Local Whisper model loaded: {whisper_model}")

            # Load audio
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logger.info(f"Transcription generated using local model")
        else:
            # Use OpenAI API
            with open(audio_file, "rb") as audio_file:
                response = await rateLimiter.api_call_with_backoff_whisper(
                    aclient.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                transcription = response

            logger.info(f"Transcription generated using OpenAI API")

        # Perform speaker diarization
        diarized_transcription = await diarize_transcription(audio_file, transcription)

        logger.info(f"Speaker diarization completed")

        performance_logs["transcription"].setdefault(f"{'local' if use_local_model else 'api'}_{whisper_model}", []).append(time.time() - start_time)
        return diarized_transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return None
    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            logger.info(f"Temporary audio file removed: {audio_file}")

    
async def diarize_transcription(audio_file_path, transcription):
    logger.info("Starting speaker diarization")
    
    # Initialize the diarization pipeline
    pipeline = pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
    
    # Move the pipeline to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    # Perform diarization
    with ProgressHook() as hook:
        diarization = pipeline(audio_file_path, hook=hook)

    # Split the transcription into sentences
    sentences = transcription.split('. ')
    
    # Load the audio file to get its duration
    audio = AudioSegment.from_file(audio_file_path)
    audio_duration = len(audio) / 1000.0  # Duration in seconds

    # Calculate the average duration of each sentence
    avg_sentence_duration = audio_duration / len(sentences)

    # Initialize variables
    current_time = 0
    speaker_turns = []
    current_speaker = None
    current_text = []

    # Assign speakers to sentences
    for sentence in sentences:
        sentence_end_time = current_time + avg_sentence_duration
        
        # Find the dominant speaker for this sentence
        speaker_counts = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= current_time and turn.end >= current_time:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + (min(turn.end, sentence_end_time) - max(turn.start, current_time))
        
        if speaker_counts:
            dominant_speaker = max(speaker_counts, key=speaker_counts.get)
        else:
            dominant_speaker = "UNKNOWN"
        
        if dominant_speaker != current_speaker:
            if current_speaker is not None:
                speaker_turns.append(f"Speaker {current_speaker}: {' '.join(current_text)}")
            current_speaker = dominant_speaker
            current_text = [sentence]
        else:
            current_text.append(sentence)
        
        current_time = sentence_end_time

    # Add the last speaker turn
    if current_text:
        speaker_turns.append(f"Speaker {current_speaker}: {' '.join(current_text)}")

    diarized_transcription = '\n'.join(speaker_turns)
    
    logger.info("Speaker diarization completed")
    return diarized_transcription

async def translate_text(text, target_languages, gpt_model, use_local_model=False):
    logger.info(f"Starting translation with model: {gpt_model}. Use local model: {use_local_model}")
    start_time = time.time()
    translations = {}
    
    if text is None:
        logger.error("Input text for translation is None")
        return None
    
    try:
        for lang in target_languages:
            if use_local_model:
                # Use local translation model
                model_name = "Helsinki-NLP/opus-mt-en-de"  # Change this for other language pairs
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                
                # Split the text into sentences, preserving speaker labels
                sentences = text.split('\n')
                translated_sentences = []
                for sentence in sentences:
                    parts = sentence.split(': ', 1)
                    if len(parts) == 2:
                        speaker, content = parts
                        inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
                        translated = model.generate(**inputs)
                        translated_content = tokenizer.decode(translated[0], skip_special_tokens=True)
                        translated_sentences.append(f"{speaker}: {translated_content}")
                    else:
                        translated_sentences.append(sentence)  # Keep non-speaker lines as is
                translation = '\n'.join(translated_sentences)
            else:
                # Use OpenAI API
                response = await rateLimiter.api_call_with_backoff(
                    aclient.chat.completions.create,
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": f"Translate the following text to {lang}. Maintain the speaker labels and format 'Speaker X: [translated text]'."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=1000
                )
                translation = response.choices[0].message.content.strip()
            translations[lang] = translation

        performance_logs["translation"].setdefault(f"{'local' if use_local_model else 'api'}_{gpt_model}", []).append(time.time() - start_time)
        return translations
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return None
    
async def analyze_sentiment(text, use_local_model=False):
    logger.info("Performing sentiment analysis")
    try:
        if use_local_model:
            # Use local sentiment analysis model
            sentiment = sentiment_analyzer(text)[0]
            return {"label": sentiment["label"], "score": sentiment["score"]}
        else:
            # Use OpenAI API for sentiment analysis
            response = await rateLimiter.api_call_with_backoff(
                aclient.chat.completions.create,
                model=current_gpt_model,
                messages=[
                    {"role": "system", "content": "Perform sentiment analysis on the following text. Respond with a JSON object containing 'label' (positive, negative, or neutral) and 'score' (between 0 and 1)."},
                    {"role": "user", "content": text}
                ]
            )
            return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return None

async def analyze_video(input_source, use_local_model=False):
    logger.info("Performing video analysis")
    try:
        video = cv2.VideoCapture(input_source)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        emotions = []
        for i in range(0, frame_count, int(fps)):  # Analyze one frame per second
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                if use_local_model:
                    emotion = emotion_detector.detect_emotions(frame)
                    if emotion:
                        emotions.append(emotion[0]['emotions'])
                else:
                    # For API-based analysis, we'd need to send the frame to an API
                    # This is a placeholder and would need to be implemented
                    pass

        video.release()
        return {
            "duration": duration,
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Error during video analysis: {str(e)}")
        return None

def save_results(whisper_model, gpt_model, result):
    model_dir = os.path.join(OUTPUT_DIR, f"{whisper_model}_{gpt_model}")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "transcription.txt"), "w") as f:
        f.write(result["transcriptions"])

    for lang, translation in result["translations"].items():
        with open(os.path.join(model_dir, f"translation_{lang}.txt"), "w") as f:
            f.write(translation)

    if result["sentiment_analysis"]:
        with open(os.path.join(model_dir, "sentiment_analysis.json"), "w") as f:
            json.dump(result["sentiment_analysis"], f, indent=2)

    if result["video_analysis"]:
        with open(os.path.join(model_dir, "video_analysis.json"), "w") as f:
            json.dump(result["video_analysis"], f, indent=2)

    with open(os.path.join(model_dir, "performance.json"), "w") as f:
        json.dump({
            "transcription_time": result["transcription_time"],
            "translation_time": result["translation_time"],
            "analysis_time": result["analysis_time"],
            "total_time": result["total_time"]
        }, f, indent=2)

def generate_performance_report(results):
    report_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
    with open(report_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Whisper Model", "GPT Model", "Transcription Time", "Translation Time", "Analysis Time", "Total Time"])
        for key, value in results.items():
            whisper_model, gpt_model = key.split("_")
            writer.writerow([
                whisper_model,
                gpt_model,
                value.get("transcription_time", "N/A"),
                value.get("translation_time", "N/A"),
                value.get("analysis_time", "N/A"),
                value.get("total_time", "N/A")
            ])
    logger.info(f"Performance report generated: {report_path}")

async def summarize_text(text, use_local_models=False):
    logger.info("Starting summarization.")
    try:
        if use_local_models:
            # Use local summarization model
            chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]  # BART models typically have a max length of 1024 tokens
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            summary = " ".join(summaries)
        else:
            # Use OpenAI API (existing code)
            max_tokens = 4000
            if num_tokens_from_string(text, current_gpt_model) > max_tokens:
                chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
                summaries = []
                for chunk in chunks:
                    response = await rateLimiter.api_call_with_backoff(
                        aclient.chat.completions.create,
                        model=current_gpt_model,
                        messages=[
                            {"role": "system", "content": "Summarize the following text concisely."},
                            {"role": "user", "content": chunk}
                        ],
                        max_tokens=500
                    )
                    summaries.append(response.choices[0].message.content.strip())
                summary = " ".join(summaries)
            else:
                response = await rateLimiter.api_call_with_backoff(
                    aclient.chat.completions.create,
                    model=current_gpt_model,
                    messages=[
                        {"role": "system", "content": "Summarize the following text concisely."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=500
                )
                summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return None
    
# At the top of your file, add:
api_semaphore = asyncio.Semaphore(5)  # Adjust this number based on your API limits

async def process_chunk(audio_chunk, video_frame=None, use_local_models=False):
    if len(audio_chunk) < 1000:
        logger.warning(f"Skipping chunk with duration {len(audio_chunk)} ms (too short)")
        return
    
    async with api_semaphore:
        start_time = time.time()
        model_key = f"{'local' if use_local_models else 'api'}_{current_gpt_model}_{current_whisper_model}"
        transcribed_text = await transcribe_audio(audio_chunk, use_local_models)
        if transcribed_text:
            conversation.add_transcription(model_key, transcribed_text)
            
            audio_features = await analyze_audio_features(audio_chunk)
            speech_features = await extract_speech_features(audio_chunk)

            if speech_features is None:
                logger.warning("Skipping detailed analysis due to insufficient speech features")
                return
            
            speech_analysis = await analyze_speech_characteristics(speech_features)
            video_emotions = await analyze_video_frame(video_frame) if video_frame is not None else None
            
            detailed_analysis_result = await detailed_analysis(
                transcribed_text, 
                audio_features, 
                speech_features,
                speech_analysis,
                video_emotions, 
                use_local_models
            )
            
            if detailed_analysis_result:
                logger.info(f"Detailed analysis: {detailed_analysis_result[:100]}...")  # Log first 100 chars

                # Add translation
                for lang in TARGET_LANGUAGES:
                    translated_text = await translate_text(detailed_analysis_result, lang, use_local_models)
                    if translated_text:
                        logger.info(f"Translated to {lang}: {translated_text[:100]}...")  # Log first 100 chars
                        conversation.add_translation(model_key, lang, translated_text)
                    else:
                        logger.warning(f"Translation to {lang} failed")
            else:
                logger.warning("Detailed analysis failed")
        else:
            logger.warning("Transcription failed for this chunk. Skipping further processing.")

        total_time = time.time() - start_time
        performance_logs["total_processing"].setdefault(model_key, []).append(total_time)
        performance_logs["transcription"].setdefault(model_key, []).append(time.time() - start_time)
        if detailed_analysis_result:
            performance_logs["analysis"].setdefault(model_key, []).append(time.time() - start_time)
        if any(translated_text for lang in TARGET_LANGUAGES):
            performance_logs["translation"].setdefault(model_key, []).append(time.time() - start_time)

        logger.debug(f"Added performance data for {model_key}")

async def chunk_producer(stream_url):
    logger.info("Starting ffmpeg process to capture audio and video.")
    process = await asyncio.create_subprocess_exec(
        'ffmpeg', '-i', stream_url, 
        '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vf', 'fps=1', '-',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    audio_buffer = b""
    frame_size = 640 * 480 * 3  # Assuming 640x480 resolution, RGB
    while True:
        try:
            chunk = await process.stdout.read(1024)
            if not chunk:
                logger.warning("No data read from ffmpeg process.")
                break

            audio_buffer += chunk
            if len(audio_buffer) > CHUNK_SIZE:
                audio_chunk = AudioSegment(
                    data=audio_buffer[:CHUNK_SIZE],
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                video_frame_data = await process.stdout.read(frame_size)
                if len(video_frame_data) == frame_size:
                    video_frame = np.frombuffer(video_frame_data, dtype=np.uint8).reshape((480, 640, 3))
                else:
                    video_frame = None
                
                await chunk_queue.put((audio_chunk, video_frame))
                audio_buffer = audio_buffer[CHUNK_SIZE:]

        except asyncio.CancelledError:
            logger.info("Task was cancelled.")
            break
        except Exception as e:
            logger.error(f"Error while processing stream: {e}")
            break

    process.terminate()
    await process.wait()
    await chunk_queue.put(None)  # Signal that production is done

async def chunk_consumer(use_local_models):
    while True:
        chunk_data = await chunk_queue.get()
        if chunk_data is None:
            break
        audio_chunk, video_frame = chunk_data
        await process_chunk(audio_chunk, video_frame, use_local_models)
        chunk_queue.task_done()

async def capture_and_process_stream(stream_url, use_local_models=False):
    producer = asyncio.create_task(chunk_producer(stream_url))
    consumers = [asyncio.create_task(chunk_consumer(use_local_models)) for _ in range(2)]  # Reduced from 5 to 3, can be increased later
    
    await producer
    await chunk_queue.join()
    for consumer in consumers:
        consumer.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)


async def process_video_file(file_path, use_local_models=False):
    logger.info(f"Processing video file: {file_path}")
    
    is_valid, error_message = validate_file_path(file_path)
    if not is_valid:
        logger.error(error_message)
        return
    
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        logger.error(f"Error opening video file: {file_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or total_frames <= 0:
        logger.error(f"Invalid video properties: FPS: {fps}, Total Frames: {total_frames}")
        video.release()
        return

    duration = total_frames / fps
    video.release()

    logger.info(f"Video properties: FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f} seconds")

    try:
        audio = AudioSegment.from_file(file_path)
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file_path}")
        return
    except Exception as e:
        logger.error(f"Error reading audio from file: {e}")
        return

    chunk_duration = 5000  # 5 seconds in milliseconds
    chunks = make_chunks(audio, chunk_duration)

    async def process_chunk_wrapper(i, chunk):
        start_time = i * chunk_duration / 1000
        end_time = min((i + 1) * chunk_duration / 1000, duration)
        
        video = cv2.VideoCapture(file_path)
        video.set(cv2.CAP_PROP_POS_MSEC, (start_time + end_time) / 2 * 1000)
        ret, frame = video.read()
        video.release()

        if ret:
            await process_chunk(chunk, frame, use_local_models)
        else:
            logger.warning(f"Could not read frame at time {(start_time + end_time) / 2:.2f} seconds")
            await process_chunk(chunk, use_local_models=use_local_models)

    semaphore = asyncio.Semaphore(3)  # Limit concurrent processed chunks

    async def semaphore_wrapper(i, chunk):
        async with semaphore:
            try:
                await process_chunk_wrapper(i, chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")

    tasks = [asyncio.create_task(semaphore_wrapper(i, chunk)) for i, chunk in enumerate(chunks)]
    
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        logger.info("Finished processing video file")

        # Ensure all tasks are done
        for task in tasks:
            if not task.done():
                logger.warning(f"Task {task} did not complete. Cancelling...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # Process any remaining data
    try:
        remaining_audio = audio[len(chunks) * chunk_duration:]
        if len(remaining_audio) > 0:
            logger.info("Processing remaining audio chunk")
            await process_chunk(remaining_audio, use_local_models=use_local_models)
    except Exception as e:
        logger.error(f"Error processing remaining audio: {e}")

    logger.info("Video processing completed")



    async def process_chunk_wrapper(i, chunk):
        start_time = i * chunk_duration / 1000
        end_time = min((i + 1) * chunk_duration / 1000, duration)
        
        video = cv2.VideoCapture(file_path)
        video.set(cv2.CAP_PROP_POS_MSEC, (start_time + end_time) / 2 * 1000)
        ret, frame = video.read()
        video.release()

        if ret:
            await process_chunk(chunk, frame, use_local_models)
        else:
            logger.warning(f"Could not read frame at time {(start_time + end_time) / 2:.2f} seconds")
            await process_chunk(chunk, use_local_models=use_local_models)

    semaphore = asyncio.Semaphore(3)  # Limit concurrent processed chunks

    async def semaphore_wrapper(i, chunk):
        async with semaphore:
            await process_chunk_wrapper(i, chunk)

    tasks = [asyncio.create_task(semaphore_wrapper(i, chunk)) for i, chunk in enumerate(chunks)]
    
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        logger.info("Finished processing video file")

        # Ensure all tasks are done
        for task in tasks:
            if not task.done():
                logger.warning(f"Task {task} did not complete. Cancelling...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

def save_performance_logs():
    os.makedirs(os.path.join(OUTPUT_DIR, "performance_logs"), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "performance_logs", f"{current_environment}_logs.json"), "w") as f:
        json.dump(performance_logs, f)

def load_performance_logs(environment):
    try:
        with open(os.path.join(OUTPUT_DIR, "performance_logs", f"{environment}_logs.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
def generate_performance_report(self):
    print('self: ', self)
    report_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
    with open(report_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GPT Model", "Whisper Model", "Processing Time (s)"])
        if isinstance(self.results, dict):  # Use results directly if it's passed as a dict
            for key, value in self.results.items():
                    gpt_model, whisper_model = key.split("_")
                    writer.writerow([gpt_model, whisper_model, value["processing_time"]])
        else:
        # Handle error or adjust logic
            print("Expected 'self.results' to be a dictionary.")

def generate_performance_plots():
    environment = current_environment
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]

    # Load data for the current environment
    all_data = {environment: load_performance_logs(environment)}

    # Plotting functions
    def plot_boxplot(data, labels, metric, title, filename, is_local):
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=data)
        plt.title(title, fontsize=16)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.xlabel("Model Configuration", fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
        if is_local:
            plt.legend(title="Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_violin(data, labels, metric, title, filename, is_local):
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=data)
        plt.title(title, fontsize=16)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.xlabel("Model Configuration", fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
        if is_local:
            plt.legend(title="Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bar(data, labels, metric, title, filename, is_local):
        means = [np.mean(d) for d in data]
        std_devs = [np.std(d) for d in data]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(data)), means, yerr=std_devs, capsize=5)
        plt.title(title, fontsize=16)
        plt.ylabel("Mean Time (seconds)", fontsize=12)
        plt.xlabel("Model Configuration", fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)
        
        if is_local:
            plt.legend(title="Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create plots directory
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    # Generate plots for each metric
    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        if all_data[environment] and metric in all_data[environment]:
            for is_local in [True, False]:
                data = []
                labels = []
                prefix = "local" if is_local else "api"
                if is_local:
                    for whisper_model in whisper_models:
                        key = f"{prefix}_{whisper_model}"
                        if key in all_data[environment][metric]:
                            data.append(all_data[environment][metric][key])
                            labels.append(f"{whisper_model}")
                else:
                    for gpt_model in gpt_models:
                        for whisper_model in whisper_models:
                            key = f"{prefix}_{gpt_model}_{whisper_model}"
                            if key in all_data[environment][metric]:
                                data.append(all_data[environment][metric][key])
                                labels.append(f"{gpt_model}\n{whisper_model}")
                
                if data:
                    title_base = f"{metric.capitalize()} Time - {environment} ({'Local' if is_local else 'API'})"
                    filename_base = f"{environment}_{metric}_{prefix}"
                    
                    model_info = f"Whisper Models: {', '.join(whisper_models)}" if is_local else f"GPT Models: {', '.join(gpt_models)} | Whisper Models: {', '.join(whisper_models)}"
                    
                    plot_boxplot(data, labels, metric, 
                                 f"{title_base} Comparison\n{model_info}", 
                                 f"{filename_base}_boxplot.png", is_local)
                    plot_violin(data, labels, metric, 
                                f"{title_base} Distribution\n{model_info}", 
                                f"{filename_base}_violin.png", is_local)
                    plot_bar(data, labels, metric, 
                             f"Mean {title_base} Comparison\n{model_info}", 
                             f"{filename_base}_bar.png", is_local)
                else:
                    logger.warning(f"No data available for plotting {metric} in {environment} ({'Local' if is_local else 'API'})")

    # Statistical analysis
    def perform_anova(data, metric, is_local):
        if len(data) < 2:
            logger.warning(f"Not enough data for ANOVA test for {metric} ({'Local' if is_local else 'API'}). Skipping.")
            return
        
        try:
            # Ensure each sublist in data has at least one element
            data = [sublist for sublist in data if len(sublist) > 0]
            if len(data) < 2:
                logger.warning(f"Not enough non-empty datasets for ANOVA test for {metric} ({'Local' if is_local else 'API'}). Skipping.")
                return
            
            f_statistic, p_value = stats.f_oneway(*data)
            with open(os.path.join(OUTPUT_DIR, "plots", f"{environment}_statistical_analysis.txt"), "a") as f:
                f.write(f"{metric.capitalize()} ANOVA Results ({'Local' if is_local else 'API'}):\n")
                f.write(f"F-statistic: {f_statistic}\n")
                f.write(f"p-value: {p_value}\n\n")
        except Exception as e:
            logger.error(f"Error performing ANOVA for {metric} ({'Local' if is_local else 'API'}): {str(e)}")

    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        if all_data[environment] and metric in all_data[environment]:
            for is_local in [True, False]:
                prefix = "local" if is_local else "api"
                data = [all_data[environment][metric][key] for key in all_data[environment][metric] if key.startswith(prefix)]
                if len(data) >= 2:
                    perform_anova(data, f"{metric}_{environment}", is_local)
                else:
                    logger.warning(f"Not enough data for ANOVA test for {metric} in {environment} ({'Local' if is_local else 'API'})")

async def run_experiment(input_source, use_local_models=False, perform_additional_analysis=False):
    global current_environment, current_gpt_model, current_whisper_model, experiment_completed
    
    whisper_models = ["tiny", "base", "small", "medium", "large"]
    gpt_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] if not use_local_models else ["local"]
    
    total_combinations = len(whisper_models) * len(gpt_models)

    results = {}

    with tqdm(total=total_combinations, desc="Experiment Progress") as pbar:
        for whisper_model in whisper_models:
            current_whisper_model = whisper_model
            
            logger.info(f"Starting experiment with Whisper model: {whisper_model}")
            
            try:
                # Transcription step
                transcription_start = time.time()
                transcriptions = await transcribe_audio(input_source, whisper_model, use_local_models)
                transcription_time = time.time() - transcription_start

                if transcriptions is None:
                    logger.error(f"Transcription failed for Whisper model: {whisper_model}")
                    continue

                for gpt_model in gpt_models:
                    current_gpt_model = gpt_model
                    
                    # Translation step
                    translation_start = time.time()
                    translations = await translate_text(transcriptions, TARGET_LANGUAGES, gpt_model, use_local_models)
                    translation_time = time.time() - translation_start

                    # Additional analysis (if requested)
                    if perform_additional_analysis:
                        analysis_start = time.time()
                        sentiment_analysis = await analyze_sentiment_per_sentence(transcriptions, use_local_models)
                        video_analysis = await analyze_video(input_source, use_local_models)
                        analysis_time = time.time() - analysis_start
                    else:
                        sentiment_analysis = None
                        video_analysis = None
                        analysis_time = 0

                    total_time = transcription_time + translation_time + analysis_time

                    results[f"{whisper_model}_{gpt_model}"] = {
                        "transcriptions": transcriptions,
                        "translations": translations,
                        "sentiment_analysis": sentiment_analysis,
                        "video_analysis": video_analysis,
                        "transcription_time": transcription_time,
                        "translation_time": translation_time,
                        "analysis_time": analysis_time,
                        "total_time": total_time
                    }

                    save_results(whisper_model, gpt_model, results[f"{whisper_model}_{gpt_model}"])
                    pbar.update(1)

            except Exception as e:
                logger.error(f"Error during experiment with Whisper model {whisper_model}: {e}", exc_info=True)
                continue
            
            logger.info(f"Finished experiment with Whisper model: {whisper_model}")
            
    logger.info(f"Experiment completed. Total results: {len(results)}")
    if not results:
        logger.warning("No results were generated during the experiment.")
    else:
        generate_performance_report(results)
    experiment_completed = True
    return results

def log_and_save_results(use_local):
    # Debug logging
    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        key = f"{'local' if use_local else 'api'}_{current_gpt_model if not use_local else ''}_{current_whisper_model}"
        if key in performance_logs[metric]:
            logger.info(f"Data points for {metric} with {key}: {len(performance_logs[metric][key])}")
        else:
            logger.warning(f"No data for {metric} with {key}")
    
    # Save intermediate results
    save_current_state()
    conversation.save_to_files()
            
def main():
    global current_environment

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Choose an environment:")
    print("1. M1 Max")
    print("2. NVIDIA 4080")
    print("3. Hetzner Cloud")
    print("4. Vultr Cloud")
    env_choice = input("Enter your choice (1-4): ")
    
    environments = ["M1 Max", "NVIDIA 4080", "Hetzner Cloud", "Vultr Cloud"]
    current_environment = environments[int(env_choice) - 1]
    
    print("\nChoose an input source:")
    print("1. Process a livestream")
    print("2. Process a video file")
    choice = input("Enter your choice (1 or 2): ")

    print("\nChoose experiment type:")
    print("1. Use local models only")
    print("2. Use API models only")
    print("3. Run both experiments")
    exp_choice = input("Enter your choice (1-3): ")

    use_local_models = exp_choice in ["1", "3"]
    use_api_models = exp_choice in ["2", "3"]

    print("\nPerform additional analysis (sentiment, video)?")
    perform_additional_analysis = input("Enter your choice (y/n): ").lower() == 'y'

    if choice == "1":
        stream_url = input("Enter the stream URL: ")
        if use_local_models:
            asyncio.run(run_experiment(stream_url, True, perform_additional_analysis))
        if use_api_models:
            asyncio.run(run_experiment(stream_url, False, perform_additional_analysis))
    elif choice == "2":
        while True:
            file_path = input("Enter the path to the video file: ")
            is_valid, error_message = validate_file_path(file_path)
            if is_valid:
                break
            print(f"Error: {error_message}")
            retry = input("Do you want to try again? (y/n): ")
            if retry.lower() != 'y':
                print("Exiting the program.")
                return
        if use_local_models:
            asyncio.run(run_experiment(file_path, True, perform_additional_analysis))
        if use_api_models:
            asyncio.run(run_experiment(file_path, False, perform_additional_analysis))
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    if experiment_completed:
        generate_performance_plots()

if __name__ == "__main__":
    main()