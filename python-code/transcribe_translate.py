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
        self.original_text = ""
        self.detailed_text = ""
        self.translations = {lang: "" for lang in TARGET_LANGUAGES}

    def add_text(self, text):
        self.original_text += text + "\n\n"

    def add_detailed_text(self, text):
        self.detailed_text += text + "\n\n"

    def add_translation(self, lang, text):
        self.translations[lang] += text + "\n\n"

    def save_to_files(self):
        os.makedirs(os.path.join(OUTPUT_DIR, "transcription"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "detailed_transcription"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "summary"), exist_ok=True)
        for lang in TARGET_LANGUAGES:
            os.makedirs(os.path.join(OUTPUT_DIR, "translations", lang), exist_ok=True)

        with open(os.path.join(OUTPUT_DIR, "transcription", "original_conversation.txt"), "w") as f:
            f.write(self.original_text)

        with open(os.path.join(OUTPUT_DIR, "detailed_transcription", "detailed_conversation.txt"), "w") as f:
            f.write(self.detailed_text)

        for lang, text in self.translations.items():
            with open(os.path.join(OUTPUT_DIR, "translations", lang, f"translated_conversation_{lang}.txt"), "w") as f:
                f.write(text)

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
        if use_local_models:
            # Perform local sentiment analysis
            sentiment = sentiment_analyzer(transcription)[0]
            analysis_result = f"Transcription: {transcription}\n"
            analysis_result += f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})\n"
            analysis_result += f"Audio Features: {audio_features}\n"
            analysis_result += f"Speech Features: {speech_features}\n"
            analysis_result += f"Speech Analysis: {speech_analysis}\n"
            analysis_result += f"Video Emotions: {video_emotions}\n"
        else:
            # Use OpenAI API
            analysis_prompt = f"""
            Analyze the following transcription, taking into account the provided audio features, speech characteristics, and video emotions:

            Tranqcription: {transcription}

            Audio Features: {audio_features}

            Speech Features: {speech_features}

            Speech Analysis: {speech_analysis}

            Video Emotions: {video_emotions}

            Based on this information:
            1. Identify the speakers.
            2. Analyze the sentiment and emotion of each sentence.
            3. Describe the speaking style, including intonation, emphasis, and overall delivery.
            4. Note any significant emotional changes or discrepancies between speech content and audio/visual cues.

            Format your response as:
            Speaker X: [Sentence] (Sentiment: [sentiment], Emotion: [emotion], Speaking Style: [description])
            """

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

        performance_logs["analysis"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return analysis_result
    except Exception as e:
        logger.error(f"Error during detailed analysis: {e}", exc_info=True)
        performance_logs["analysis"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return transcription


async def transcribe_audio(audio_chunk, use_local_model=False):
    logger.info(f"Starting transcription. Use local model: {use_local_model}")
    start_time = time.time()
    try:
        if use_local_model:
            # Use local Whisper model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            logger.info(f"Loading Whisper model: openai/whisper-{current_whisper_model}")
            
            model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{current_whisper_model}")
            model.to(device)
            processor = WhisperProcessor.from_pretrained(f"openai/whisper-{current_whisper_model}")
            
            # Convert audio chunk to numpy array
            audio_array = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
            
            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Resample audio to 16kHz if necessary
            if audio_chunk.frame_rate != 16000:
                logger.info(f"Resampling audio from {audio_chunk.frame_rate}Hz to 16000Hz")
                audio_array = librosa.resample(audio_array, orig_sr=audio_chunk.frame_rate, target_sr=16000)
            
            # Process audio with the Whisper processor
            input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Generate transcription
            logger.info("Generating transcription")
            generated_ids = model.generate(input_features)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcription result: {transcription[:100]}...")  # Log first 100 chars
        else:
            # Use OpenAI API
            with io.BytesIO() as audio_file:
                audio_chunk.export(audio_file, format="mp3")
                audio_file.seek(0)
                try:
                    response = await rateLimiter.api_call_with_backoff_whisper(
                        aclient.audio.transcriptions.create,
                        model="whisper-1",
                        file=("audio.mp3", audio_file),
                        response_format="text"
                    )
                    transcription = response
                except Exception as e:
                    logger.error(f"Error during API transcription: {str(e)}")
                    return None

        performance_logs["transcription"].setdefault(f"{'local' if use_local_model else 'api'}_{current_whisper_model}", []).append(time.time() - start_time)
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        logger.error(f"Audio chunk details: Duration: {len(audio_chunk) / 1000}s, Frame rate: {audio_chunk.frame_rate}, Channels: {audio_chunk.channels}")
        performance_logs["transcription"].setdefault(f"{'local' if use_local_model else 'api'}_{current_whisper_model}", []).append(time.time() - start_time)
        return None
    
async def translate_text(text, target_lang, use_local_model=False):
    if not text:
        logger.warning("Empty text provided for translation. Skipping.")
        return None
    
    logger.info(f"Starting translation to {target_lang}.")
    start_time = time.time()
    try:
        if use_local_model:
            # Use local translation model
            model_name = "Helsinki-NLP/opus-mt-en-de"  # Changed from 'ger' to 'de'
            
            # Use the token if available
            if HF_TOKEN:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            # Use OpenAI API
            response = await rateLimiter.api_call_with_backoff(
                aclient.chat.completions.create,
                model=current_gpt_model,
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_lang}. Maintain the speaker labels if present."},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000
            )
            translation = response.choices[0].message.content.strip()
        performance_logs["translation"].setdefault(f"{'local' if use_local_model else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return translation
    except Exception as e:
        logger.error(f"Error during translation to {target_lang}: {e}", exc_info=True)
        performance_logs["translation"].setdefault(f"{'local' if use_local_model else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return None

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
        transcribed_text = await transcribe_audio(audio_chunk, use_local_models)
        if transcribed_text:
            conversation.add_text(transcribed_text)
            
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
                conversation.add_detailed_text(detailed_analysis_result)

                # Add translation
                for lang in TARGET_LANGUAGES:
                    translated_text = await translate_text(detailed_analysis_result, lang, use_local_models)
                    if translated_text:
                        logger.info(f"Translated to {lang}: {translated_text[:100]}...")  # Log first 100 chars
                        conversation.add_translation(lang, translated_text)
                    else:
                        logger.warning(f"Translation to {lang} failed")
            else:
                logger.warning("Detailed analysis failed")
        else:
            logger.warning("Transcription failed for this chunk. Skipping further processing.")

        total_time = time.time() - start_time
        key = f"{'local' if use_local_models else 'api'}_{current_gpt_model}_{current_whisper_model}"
        performance_logs["total_processing"].setdefault(key, []).append(total_time)
        performance_logs["transcription"].setdefault(key, []).append(time.time() - start_time)
        if detailed_analysis_result:
            performance_logs["analysis"].setdefault(key, []).append(time.time() - start_time)
        if any(translated_text for lang in TARGET_LANGUAGES):
            performance_logs["translation"].setdefault(key, []).append(time.time() - start_time)

        logger.debug(f"Added performance data for {key}")

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
    
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"File does not exist: {file_path}")
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
        await asyncio.gather(*tasks, return_exceptions=True)
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
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task} could not be cancelled within timeout.")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task {task}: {e}")

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
            try:
                await process_chunk_wrapper(i, chunk)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")

    tasks = [asyncio.create_task(semaphore_wrapper(i, chunk)) for i, chunk in enumerate(chunks)]
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
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
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task} could not be cancelled within timeout.")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task {task}: {e}")

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

def generate_performance_plots():
    environment = current_environment
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]

    # Load data for the current environment
    all_data = {environment: load_performance_logs(environment)}

    # Plotting functions
    def plot_boxplot(data, labels, metric, title, filename):
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=data)
        plt.title(title, fontsize=16)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.xlabel("Model Configuration", fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
        plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_violin(data, labels, metric, title, filename):
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=data)
        plt.title(title, fontsize=16)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.xlabel("Model Configuration", fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
        plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bar(data, labels, metric, title, filename):
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
        
        plt.legend(title="GPT Model - Whisper Model", title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create plots directory
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    # Generate plots for each metric
    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        if all_data[environment] and metric in all_data[environment]:
            data = []
            labels = []
            for gpt_model in gpt_models:
                for whisper_model in whisper_models:
                    key = f"api_{gpt_model}_{whisper_model}"
                    if key in all_data[environment][metric]:
                        data.append(all_data[environment][metric][key])
                        labels.append(f"{gpt_model}\n{whisper_model}")
            
            if data:
                title_base = f"{metric.capitalize()} Time - {environment}"
                filename_base = f"{environment}_{metric}"
                
                plot_boxplot(data, labels, metric, 
                             f"{title_base} Comparison\nGPT Models: {', '.join(gpt_models)} | Whisper Models: {', '.join(whisper_models)}", 
                             f"{filename_base}_boxplot.png")
                plot_violin(data, labels, metric, 
                            f"{title_base} Distribution\nGPT Models: {', '.join(gpt_models)} | Whisper Models: {', '.join(whisper_models)}", 
                            f"{filename_base}_violin.png")
                plot_bar(data, labels, metric, 
                         f"Mean {title_base} Comparison\nGPT Models: {', '.join(gpt_models)} | Whisper Models: {', '.join(whisper_models)}", 
                         f"{filename_base}_bar.png")
            else:
                logger.warning(f"No data available for plotting {metric} in {environment}")

    # Statistical analysis
    def perform_anova(data, metric):
        if len(data) < 2:
            logger.warning(f"Not enough data for ANOVA test for {metric}. Skipping.")
            return
        
        try:
            # Ensure each sublist in data has at least one element
            data = [sublist for sublist in data if len(sublist) > 0]
            if len(data) < 2:
                logger.warning(f"Not enough non-empty datasets for ANOVA test for {metric}. Skipping.")
                return
            
            f_statistic, p_value = stats.f_oneway(*data)
            with open(os.path.join(OUTPUT_DIR, "plots", f"{environment}_statistical_analysis.txt"), "a") as f:
                f.write(f"{metric.capitalize()} ANOVA Results:\n")
                f.write(f"F-statistic: {f_statistic}\n")
                f.write(f"p-value: {p_value}\n\n")
        except Exception as e:
            logger.error(f"Error performing ANOVA for {metric}: {str(e)}")

    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        if all_data[environment] and metric in all_data[environment]:
            data = [all_data[environment][metric][key] for key in all_data[environment][metric] if key.startswith("api_")]
            if len(data) >= 2:
                perform_anova(data, f"{metric}_{environment}")
            else:
                logger.warning(f"Not enough data for ANOVA test for {metric} in {environment}")

async def run_experiment(input_source, use_local_models=False):
    global current_environment, current_gpt_model, current_whisper_model, experiment_completed
    
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]
    
    total_combinations = len(gpt_models) * len(whisper_models)
    
    with tqdm(total=total_combinations, desc="Experiment Progress") as pbar:
        for gpt_model in gpt_models:
            for whisper_model in whisper_models:
                current_gpt_model = gpt_model
                current_whisper_model = whisper_model
                
                logger.info(f"Starting experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}")
                
                try:
                    if isinstance(input_source, str) and input_source.startswith("rtmp://"):
                        await capture_and_process_stream(input_source, use_local_models)
                    else:
                        await process_video_file(input_source, use_local_models)
                except Exception as e:
                    logger.error(f"Error during experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}: {e}", exc_info=True)
                    continue  # Continue with the next model combination
                
                logger.info(f"Finished experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}")
                
                # Debug logging
                for metric in ["transcription", "translation", "analysis", "total_processing"]:
                    key = f"{'local' if use_local_models else 'api'}_{gpt_model}_{whisper_model}"
                    if key in performance_logs[metric]:
                        logger.info(f"Data points for {metric} with {key}: {len(performance_logs[metric][key])}")
                    else:
                        logger.warning(f"No data for {metric} with {key}")
                
                # Save intermediate results after each model combination
                save_current_state()
                
                # Update progress bar
                pbar.update(1)
    
    try:
        summary = await summarize_text(conversation.original_text, use_local_models)
        if summary:
            print(f"Summary: {summary}")
            with open(os.path.join(OUTPUT_DIR, "summary", "conversation_summary.txt"), "w") as f:
                f.write(summary)
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
    
    logger.info("Experiment run completed.")
    experiment_completed = True
            
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

    use_local_models = input("Use local models? (y/n): ").lower() == 'y'

    if choice == "1":
        stream_url = input("Enter the stream URL: ")
        asyncio.run(run_experiment(stream_url, use_local_models))
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
        asyncio.run(run_experiment(file_path, use_local_models))
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    if experiment_completed:
        generate_performance_plots()

if __name__ == "__main__":
    main()