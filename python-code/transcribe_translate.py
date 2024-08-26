import os
import asyncio
import logging
import time
import cv2
import csv
import numpy as np
from fer import FER
import librosa
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
import tiktoken
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from scipy import stats
import signal
import sys
import rateLimiter
import parselmouth
import pandas as pd
from parselmouth.praat import call
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
import soundfile as sf
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import detect_silence
from dotenv import load_dotenv

# Environment setup
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 16000 * 10 * 2  # 10 seconds of audio at 16kHz, 16-bit
TARGET_LANGUAGES = ['ger']  # Only German for testing
OUTPUT_DIR = "output"
MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB, just under OpenAI's 26 MB limit

# Global variables
current_environment = None
current_gpt_model = None
current_whisper_model = None
experiment_completed = False

# Performance logging
performance_logs = {
    "transcription": {},
    "translation": {},
    "analysis": {},
    "speaker_diarization": {},
    "sentiment_analysis": {},
    "total_processing": {}
}

# Cost tracking
cost_logs = {
    "transcription": {},
    "translation": {},
    "analysis": {},
    "total": {}
}

# Initialize AsyncOpenAI client
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Rate limiting
rate_limit = AsyncLimiter(10, 60)  # 10 requests per minute

# Queue for chunk processing
chunk_queue = asyncio.Queue()

# Model caching
model_cache = {}

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Initialize FER
emotion_detector = FER(mtcnn=True)

class Experiment:
    def __init__(self, input_source, use_local_models, perform_additional_analysis, environment):
        self.input_source = input_source
        self.use_local_models = use_local_models
        self.perform_additional_analysis = perform_additional_analysis
        self.environment = environment
        self.results = {}

    async def run(self):
        global current_gpt_model, current_whisper_model, experiment_completed

        whisper_models = ["tiny", "base", "small", "medium", "large"]
        gpt_models = ["local"] if self.use_local_models else ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

        for whisper_model in whisper_models:
            current_whisper_model = whisper_model
            for gpt_model in gpt_models:
                current_gpt_model = gpt_model
                
                key = f"{whisper_model}_{gpt_model}"
                result = await self.process_video(whisper_model, gpt_model)
                self.results[key] = result

        experiment_completed = True
        self.compare_results()
        self.save_results()
        self.generate_performance_report()
        self.generate_performance_plots()

    async def process_video(self, whisper_model, gpt_model):
        start_time = time.time()
        
        # Transcription
        transcription = await transcribe_audio(self.input_source, whisper_model, self.use_local_models)
        
        # Speaker Diarization
        diarized_transcription = await perform_speaker_diarization(self.input_source, transcription, self.use_local_models)
        
        # Translation
        translations = await translate_text(diarized_transcription, TARGET_LANGUAGES, gpt_model, self.use_local_models)
        
        # Sentiment Analysis
        sentiment_analysis = await analyze_sentiment(diarized_transcription, self.use_local_models)
        
        # Video Emotion Analysis
        video_emotions = await analyze_video_emotions(self.input_source, self.use_local_models)
        
        total_time = time.time() - start_time
        
        return {
            "transcription": diarized_transcription,
            "translations": translations,
            "sentiment_analysis": sentiment_analysis,
            "video_emotions": video_emotions,
            "processing_time": total_time,
            "total_time": total_time
        }

    def compare_results(self):
        logger.info("Comparing results across models")
        if not self.results:
            logger.warning("No results to compare")
            return
        
        best_model = min(self.results, key=lambda x: self.results[x].get("total_time", float('inf')))
        logger.info(f"Best performing model: {best_model}")

    def save_results(self):
        with open(f"results_{self.environment}.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def generate_performance_report(self):
        report_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
        with open(report_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Whisper Model", "GPT Model", "Transcription Time", "Translation Time", "Analysis Time", "Total Time", "Estimated Cost"])
            for key, value in self.results.items():
                whisper_model, gpt_model = key.split("_")
                writer.writerow([
                    whisper_model,
                    gpt_model,
                    value.get("transcription_time", "N/A"),
                    value.get("translation_time", "N/A"),
                    value.get("analysis_time", "N/A"),
                    value.get("total_time", "N/A"),
                    value.get("estimated_cost", "N/A")
                ])
        logger.info(f"Performance report generated: {report_path}")

    def generate_performance_plots(self):
        # Implementation of plot generation
        pass

async def transcribe_audio(input_source, whisper_model, use_local_model):
    logger.info(f"Starting transcription with Whisper model: {whisper_model}")
    start_time = time.time()
    
    try:
        # Extract and preprocess audio
        audio_file = preprocess_audio(input_source)

        if use_local_model:
            transcription = await local_transcribe(audio_file, whisper_model)
        else:
            transcription = await api_transcribe(audio_file)

        transcription_time = time.time() - start_time
        performance_logs["transcription"][f"{'local' if use_local_model else 'api'}_{whisper_model}"] = transcription_time
        
        # Estimate cost for API usage
        if not use_local_model:
            cost = estimate_cost("whisper-1", len(audio_file) / 1000)  # Assuming audio length in seconds
            cost_logs["transcription"][f"api_{whisper_model}"] = cost
        
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return None

async def local_transcribe(audio_file, whisper_model):
    model = get_local_model(f"whisper-{whisper_model}")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model}")
    
    audio, sr = sf.read(audio_file)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

async def api_transcribe(audio_file):
    with open(audio_file, "rb") as audio_file:
        response = await rateLimiter.api_call_with_backoff_whisper(
            aclient.audio.transcriptions.create,
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return response

async def perform_speaker_diarization(audio_file, transcription, use_local_model):
    logger.info("Performing speaker diarization")
    try:
        if use_local_model:
            return await local_speaker_diarization(audio_file, transcription)
        else:
            return await api_speaker_diarization(audio_file, transcription)
    except Exception as e:
        logger.error(f"Error during speaker diarization: {str(e)}")
        return transcription

async def local_speaker_diarization(audio_file, transcription):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=HF_TOKEN)
    pipeline = pipeline.to(device)
    diarization = pipeline(audio_file)
    
    # Implement the logic to align diarization with transcription
    sentences = transcription.split('. ')
    diarized_transcription = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_sentences = [s for s in sentences if turn.start <= sentences.index(s)/len(sentences)*turn.end < turn.end]
        if turn_sentences:
            diarized_transcription.extend([f"Speaker {speaker}: {s}" for s in turn_sentences])
    
    return '\n'.join(diarized_transcription)

async def api_speaker_diarization(audio_file, transcription):
    # Implement API-based speaker diarization
    # You may need to use a service that provides this functionality
    logger.warning("API-based speaker diarization not implemented. Returning original transcription.")
    return transcription

async def translate_text(text, target_languages, gpt_model, use_local_model):
    logger.info(f"Starting translation with model: {gpt_model}")
    start_time = time.time()
    
    translations = {}
    try:
        for lang in target_languages:
            if use_local_model:
                translation = await local_translate(text, lang)
            else:
                translation = await api_translate(text, lang, gpt_model)
            translations[lang] = translation

        translation_time = time.time() - start_time
        performance_logs["translation"][f"{'local' if use_local_model else 'api'}_{gpt_model}_{current_whisper_model}"] = translation_time
        
        # Estimate cost for API usage
        if not use_local_model:
            cost = estimate_cost(gpt_model, num_tokens_from_string(text, gpt_model))
            cost_logs["translation"][f"api_{gpt_model}"] = cost
        
        return translations
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return None

async def local_translate(text, target_language):
    model_name = "Helsinki-NLP/opus-mt-en-de"  # Change this for other language pairs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    sentences = text.split('\n')
    translated_sentences = []
    for sentence in sentences:
        parts = sentence.split(': ', 1)
        if len(parts) == 2:
            speaker, content = parts
            inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512).to(device)
            translated = model.generate(**inputs)
            translated_content = tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_sentences.append(f"{speaker}: {translated_content}")
        else:
            translated_sentences.append(sentence)
    return '\n'.join(translated_sentences)

async def api_translate(text, target_language, gpt_model):
    response = await rateLimiter.api_call_with_backoff(
        aclient.chat.completions.create,
        model=gpt_model,
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}. Maintain the speaker labels and format 'Speaker X: [translated text]'."},
            {"role": "user", "content": text}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

async def analyze_sentiment(text, use_local_model):
    logger.info("Performing sentiment analysis")
    try:
        if use_local_model:
            return await local_sentiment_analysis(text)
        else:
            return await api_sentiment_analysis(text)
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        return None

async def local_sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    sentiments = []
    for line in text.split('\n'):
        parts = line.split(': ', 1)
        if len(parts) == 2:
            speaker, sentence = parts
            sentiment = sentiment_analyzer(sentence)[0]
            sentiments.append({
                "speaker": speaker,
                "sentence": sentence,
                "sentiment": {"label": sentiment["label"], "score": sentiment["score"]}
            })
    return sentiments

async def api_sentiment_analysis(text):
    response = await rateLimiter.api_call_with_backoff(
        aclient.chat.completions.create,
        model=current_gpt_model,
        messages=[
            {"role": "system", "content": "Perform sentiment analysis on the following text. For each line, respond with a JSON object containing 'speaker', 'sentence', and 'sentiment' (with 'label' and 'score')."},
            {"role": "user", "content": text}
        ]
    )
    return json.loads(response.choices[0].message.content)

async def analyze_video_emotions(input_source, use_local_model):
    logger.info("Performing video emotion analysis")
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
                    # Implement API-based emotion detection if available
                    pass

        video.release()
        return {
            "duration": duration,
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Error during video emotion analysis: {str(e)}")
        return None

def preprocess_audio(input_source):
    video = VideoFileClip(input_source)
    audio = video.audio
    audio_file = input_source.rsplit('.', 1)[0] + '_temp.wav'
    audio.write_audiofile(audio_file, codec='pcm_s16le')
    video.close()

    # Resample audio to 16000 Hz
    audio, sr = librosa.load(audio_file, sr=16000)
    sf.write(audio_file, audio, sr, subtype='PCM_16')

    return audio_file

def get_local_model(model_name):
    if model_name not in model_cache:
        if "whisper" in model_name:
            model_cache[model_name] = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        elif "sentiment" in model_name:
            model_cache[model_name] = pipeline("sentiment-analysis", device=device)
    return model_cache[model_name]

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def estimate_cost(model, usage):
    # Define cost per unit for different models
    costs = {
        "whisper-1": 0.006 / 60,  # $0.006 per minute
        "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
        "gpt-4-turbo": 0.01 / 1000,  # $0.01 per 1K tokens
    }
    
    if model in costs:
        return costs[model] * usage
    else:
        logger.warning(f"Unknown model for cost estimation: {model}")
        return 0

async def process_livestream(stream_url, use_local_models, perform_additional_analysis, environment):
    logger.info(f"Processing livestream: {stream_url}")
    try:
        global current_gpt_model, current_whisper_model
        current_whisper_model = "large"
        current_gpt_model = "local" if use_local_models else "gpt-3.5-turbo"

        await capture_and_process_stream(stream_url, use_local_models)

        if perform_additional_analysis:
            # Perform additional analysis on processed chunks
            processed_chunks = conversation.transcriptions.get(f"{'local' if use_local_models else 'api'}_{current_gpt_model}_{current_whisper_model}", [])
            
            for chunk in processed_chunks:
                sentiment_analysis = await analyze_sentiment(chunk, use_local_models)
                logger.info(f"Sentiment analysis for chunk: {sentiment_analysis}")

        logger.info("Livestream processing completed")
    except Exception as e:
        logger.error(f"Error processing livestream: {e}")

async def capture_and_process_stream(stream_url, use_local_models):
    producer = asyncio.create_task(chunk_producer(stream_url))
    consumers = [asyncio.create_task(chunk_consumer(use_local_models)) for _ in range(3)]
    
    await producer
    await chunk_queue.join()
    for consumer in consumers:
        consumer.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)

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

async def process_chunk(audio_chunk, video_frame, use_local_models):
    if len(audio_chunk) < 1000:
        logger.warning(f"Skipping chunk with duration {len(audio_chunk)} ms (too short)")
        return
    
    async with asyncio.Semaphore(5):  # Limit concurrent API calls
        start_time = time.time()
        model_key = f"{'local' if use_local_models else 'api'}_{current_gpt_model}_{current_whisper_model}"
        transcribed_text = await transcribe_audio(audio_chunk, current_whisper_model, use_local_models)
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
                logger.info(f"Detailed analysis: {detailed_analysis_result[:100]}...")

                for lang in TARGET_LANGUAGES:
                    translated_text = await translate_text(detailed_analysis_result, [lang], current_gpt_model, use_local_models)
                    if translated_text:
                        logger.info(f"Translated to {lang}: {translated_text[lang][:100]}...")
                        conversation.add_translation(model_key, lang, translated_text[lang])
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

def generate_performance_plots():
    environment = current_environment
    gpt_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    whisper_models = ["tiny", "base", "small", "medium", "large"]

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
                                data.append(all_data[environment][metric][key])labels.append(f"{gpt_model}\n{whisper_model}")
                
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
                        sentiment_analysis = await analyze_sentiment(transcriptions, use_local_models)
                        video_analysis = await analyze_video_emotions(input_source, use_local_models)
                        analysis_time = time.time() - analysis_start
                    else:
                        sentiment_analysis = None
                        video_analysis = None
                        analysis_time = 0

                    total_time = transcription_time + translation_time + analysis_time

                    # Estimate cost
                    estimated_cost = 0
                    if not use_local_models:
                        estimated_cost += estimate_cost("whisper-1", len(input_source) / 1000)  # Assuming audio length in seconds
                        estimated_cost += estimate_cost(gpt_model, num_tokens_from_string(transcriptions, gpt_model))
                        if perform_additional_analysis:
                            estimated_cost += estimate_cost(gpt_model, num_tokens_from_string(json.dumps(sentiment_analysis), gpt_model))

                    results[f"{whisper_model}_{gpt_model}"] = {
                        "transcriptions": transcriptions,
                        "translations": translations,
                        "sentiment_analysis": sentiment_analysis,
                        "video_analysis": video_analysis,
                        "transcription_time": transcription_time,
                        "translation_time": translation_time,
                        "analysis_time": analysis_time,
                        "total_time": total_time,
                        "estimated_cost": estimated_cost
                    }

                    pbar.update(1)

            except Exception as e:
                logger.error(f"Error during experiment with Whisper model {whisper_model}: {e}", exc_info=True)
                continue
            
            logger.info(f"Finished experiment with Whisper model: {whisper_model}")
            
    logger.info(f"Experiment completed. Total results: {len(results)}")
    if not results:
        logger.warning("No results were generated during the experiment.")
    else:
        save_results(results)
        generate_performance_report(results)
        generate_performance_plots()
    experiment_completed = True
    return results

def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"results_{current_environment}.json"), "w") as f:
        json.dump(results, f, indent=2)

def generate_performance_report(results):
    report_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
    with open(report_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Environment", "Whisper Model", "GPT Model", "Transcription Time", "Translation Time", "Analysis Time", "Total Time", "Estimated Cost"])
        for key, value in results.items():
            whisper_model, gpt_model = key.split("_")
            writer.writerow([
                current_environment,
                whisper_model,
                gpt_model,
                value.get("transcription_time", "N/A"),
                value.get("translation_time", "N/A"),
                value.get("analysis_time", "N/A"),
                value.get("total_time", "N/A"),
                value.get("estimated_cost", "N/A")
            ])
    logger.info(f"Performance report generated: {report_path}")

async def main():
    global current_environment

    # Environment selection
    environments = ["M1 Max", "NVIDIA 4070", "Vultr Cloud", "Hetzner Cloud"]
    print("Choose an environment:")
    for i, env in enumerate(environments, 1):
        print(f"{i}. {env}")
    env_choice = int(input("Enter your choice (1-4): ")) - 1
    current_environment = environments[env_choice]

    # Input source selection
    print("\nChoose an input source:")
    print("1. Process a video file")
    print("2. Process a livestream")
    source_choice = input("Enter your choice (1 or 2): ")

    # Experiment type selection
    print("\nChoose experiment type:")
    print("1. Use local models only")
    print("2. Use API models only")
    print("3. Run both experiments")
    exp_choice = input("Enter your choice (1-3): ")

    use_local_models = exp_choice in ["1", "3"]
    use_api_models = exp_choice in ["2", "3"]

    # Additional analysis option
    perform_additional_analysis = input("\nPerform additional analysis (sentiment, video)? (y/n): ").lower() == 'y'

    try:
        if source_choice == "1":
            input_source = input("Enter the path to the video file: ")
            if use_local_models:
                await run_experiment(input_source, True, perform_additional_analysis)
            if use_api_models:
                await run_experiment(input_source, False, perform_additional_analysis)
        else:
            stream_url = input("Enter the livestream URL: ")
            if use_local_models:
                await process_livestream(stream_url, True, perform_additional_analysis, current_environment)
            if use_api_models:
                await process_livestream(stream_url, False, perform_additional_analysis, current_environment)

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
    finally:
        logger.info("Experiment completed")
        save_performance_logs()
        generate_performance_plots()

if __name__ == "__main__":
    asyncio.run(main())