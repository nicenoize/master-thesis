import os
import asyncio
import logging
import io
import time
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks
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

CHUNK_SIZE = 16000 * 5 * 2  # 5 seconds of audio at 16kHz, 16-bit
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
current_whisper_model = "base"

class Conversation:
    def __init__(self):
        self.original_text = ""
        self.translations = {lang: "" for lang in TARGET_LANGUAGES}

    def add_text(self, text):
        self.original_text += text + "\n\n"

    def add_translation(self, lang, text):
        self.translations[lang] += text + "\n\n"

    def save_to_files(self):
        os.makedirs(os.path.join(OUTPUT_DIR, "transcription"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "summary"), exist_ok=True)
        for lang in TARGET_LANGUAGES:
            os.makedirs(os.path.join(OUTPUT_DIR, "translations", lang), exist_ok=True)

        with open(os.path.join(OUTPUT_DIR, "transcription", "original_conversation.txt"), "w") as f:
            f.write(self.original_text)

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
async def api_call_with_backoff(func, *args, **kwargs):
    try:
        async with rate_limit:
            return await func(*args, **kwargs)
    except Exception as e:
        if "Too Many Requests" in str(e):
            logger.warning("Rate limit exceeded. Retrying with exponential backoff.")
            raise tenacity.TryAgain
        raise

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

sentiment_analyzer = pipeline("sentiment-analysis")

async def detailed_analysis(transcription, audio_features, video_emotions, use_local_models=False):
    logger.info("Performing detailed analysis.")
    start_time = time.time()
    try:
        if use_local_models:
            # Perform local sentiment analysis
            sentiment = sentiment_analyzer(transcription)[0]
            analysis_result = f"Transcription: {transcription}\n"
            analysis_result += f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})\n"
            analysis_result += f"Audio Features: MFCCs and Chroma data available\n"
            analysis_result += f"Video Emotions: {video_emotions}\n"
        else:
            # Use OpenAI API
            analysis_prompt = f"""
            Analyze the following transcription, taking into account the provided audio features and video emotions:

            Transcription: {transcription}

            Audio Features:
            MFCCs: {audio_features['mfccs']}
            Chroma: {audio_features['chroma']}

            Video Emotions: {video_emotions}

            Based on this information:
            1. Identify the speakers.
            2. Analyze the sentiment of each sentence.
            3. Describe the intonation and overall vibe of each speaker's delivery.
            4. Note any significant emotional changes or discrepancies between speech content and audio/visual cues.

            Format your response as:
            Speaker X: [Sentence] (Sentiment: [sentiment], Intonation: [description], Vibe: [description])
            """

            response = await api_call_with_backoff(
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
        logger.error(f"Error during detailed analysis: {e}")
        performance_logs["analysis"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return transcription
    
async def transcribe_audio(audio_chunk, use_local_model=False):
    logger.info("Starting transcription.")
    start_time = time.time()
    try:
        if use_local_model:
            # Use local Whisper model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{current_whisper_model}").to(device)
            processor = WhisperProcessor.from_pretrained(f"openai/whisper-{current_whisper_model}")
            
            # Convert audio chunk to numpy array
            audio_array = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
            
            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Resample audio to 16kHz if necessary
            if audio_chunk.frame_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=audio_chunk.frame_rate, target_sr=16000)
            
            # Process audio with the Whisper processor
            input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Generate transcription
            generated_ids = model.generate(input_features, language="en", task="transcribe")
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # Use OpenAI API
            with io.BytesIO() as audio_file:
                audio_chunk.export(audio_file, format="mp3")
                audio_file.seek(0)
                response = await api_call_with_backoff(
                    aclient.audio.transcriptions.create,
                    model="whisper-1",
                    file=("audio.mp3", audio_file),
                    response_format="text"
                )
            transcription = response

        performance_logs["transcription"].setdefault(f"{'local' if use_local_model else 'api'}_{current_whisper_model}", []).append(time.time() - start_time)
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        logger.error(f"Audio chunk details: Duration: {len(audio_chunk) / 1000}s, Frame rate: {audio_chunk.frame_rate}, Channels: {audio_chunk.channels}")
        performance_logs["transcription"].setdefault(f"{'local' if use_local_model else 'api'}_{current_whisper_model}", []).append(time.time() - start_time)
        return None
    
async def translate_text(text, target_lang, use_local_model=False):
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
            response = await api_call_with_backoff(
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
        logger.error(f"Error during translation to {target_lang}: {e}")
        performance_logs["translation"].setdefault(f"{'local' if use_local_model else 'api'}_{current_gpt_model}", []).append(time.time() - start_time)
        return None

async def summarize_text(text):
    logger.info("Starting summarization.")
    try:
        max_tokens = 4000
        if num_tokens_from_string(text, current_gpt_model) > max_tokens:
            chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
            summaries = []
            for chunk in chunks:
                response = await api_call_with_backoff(
                    aclient.chat.completions.create,
                    model=current_gpt_model,
                    messages=[
                        {"role": "system", "content": "Summarize the following text concisely."},
                        {"role": "user", "content": chunk}
                    ],
                    max_tokens=500
                )
                summaries.append(response.choices[0].message.content.strip())
            return " ".join(summaries)
        else:
            response = await api_call_with_backoff(
                aclient.chat.completions.create,
                model=current_gpt_model,
                messages=[
                    {"role": "system", "content": "Summarize the following text concisely."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return None

async def process_chunk(audio_chunk, video_frame=None, use_local_models=False):
    start_time = time.time()
    transcribed_text = await transcribe_audio(audio_chunk, use_local_models)
    if transcribed_text:
        audio_features = await analyze_audio_features(audio_chunk)
        video_emotions = await analyze_video_frame(video_frame) if video_frame is not None else None
        detailed_analysis_result = await detailed_analysis(transcribed_text, audio_features, video_emotions, use_local_models)
        logger.info(f"Detailed analysis: {detailed_analysis_result}")
        conversation.add_text(detailed_analysis_result)

        translation_tasks = [translate_text(detailed_analysis_result, lang, use_local_models) for lang in TARGET_LANGUAGES]
        translations = await asyncio.gather(*translation_tasks)

        for lang, translation in zip(TARGET_LANGUAGES, translations):
            if translation:
                logger.info(f"Translated ({lang}): {translation}")
                conversation.add_translation(lang, translation)
    else:
        logger.warning("Transcription failed for this chunk. Skipping further processing.")

    total_time = time.time() - start_time
    performance_logs["total_processing"].setdefault(f"{'local' if use_local_models else 'api'}_{current_gpt_model}_{current_whisper_model}", []).append(total_time)

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
    consumers = [asyncio.create_task(chunk_consumer(use_local_models)) for _ in range(5)]  # Create 5 consumers
    
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

    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing to 5 chunks

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
    environments = ["M1 Max", "NVIDIA 4080", "Hetzner Cloud", "Vultr Cloud"]
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]

    # Load data for all environments
    all_data = {env: load_performance_logs(env) for env in environments}

    # Plotting functions
    def plot_boxplot(data, metric, title, filename):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_violin(data, metric, title, filename):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=data)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bar(data, metric, title, filename):
        means = [np.mean(d) for d in data]
        std_devs = [np.std(d) for d in data]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(data)), means, yerr=std_devs, capsize=5)
        plt.title(title)
        plt.ylabel("Mean Time (seconds)")
        plt.xlabel("Configuration")
        plt.xticks(range(len(data)), [f"Config {i+1}" for i in range(len(data))], rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", filename), dpi=300, bbox_inches="tight")
        plt.close()

    # Create plots directory
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    # Generate plots for each metric
    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        for env in environments:
            if all_data[env]:
                data = []
                labels = []
                for gpt_model in gpt_models:
                    for whisper_model in whisper_models:
                        key = f"api_{gpt_model}_{whisper_model}"
                        if key in all_data[env][metric]:
                            data.append(all_data[env][metric][key])
                            labels.append(f"{env}\n{gpt_model}\n{whisper_model}")
                
                if data:
                    plot_boxplot(data, metric, f"{metric.capitalize()} Time Comparison - {env}", f"{env}_{metric}_boxplot.png")
                    plot_violin(data, metric, f"{metric.capitalize()} Time Distribution - {env}", f"{env}_{metric}_violin.png")
                    plot_bar(data, metric, f"Mean {metric.capitalize()} Time Comparison - {env}", f"{env}_{metric}_bar.png")

    # Statistical analysis
    def perform_anova(data, metric):
        f_statistic, p_value = stats.f_oneway(*data)
        with open(os.path.join(OUTPUT_DIR, "plots", "statistical_analysis.txt"), "a") as f:
            f.write(f"{metric.capitalize()} ANOVA Results:\n")
            f.write(f"F-statistic: {f_statistic}\n")
            f.write(f"p-value: {p_value}\n\n")

    for metric in ["transcription", "translation", "analysis", "total_processing"]:
        for env in environments:
            if all_data[env]:
                data = [all_data[env][metric][key] for key in all_data[env][metric] if key.startswith("api_")]
                if data:
                    perform_anova(data, f"{metric}_{env}")

async def run_experiment(input_source, use_local_models=False):
    global current_environment, current_gpt_model, current_whisper_model, experiment_completed
    
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]
    
    try:
        for gpt_model in gpt_models:
            for whisper_model in whisper_models:
                current_gpt_model = gpt_model
                current_whisper_model = whisper_model
                
                logger.info(f"Starting experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}")
                
                if isinstance(input_source, str) and input_source.startswith("rtmp://"):
                    await capture_and_process_stream(input_source, use_local_models)
                else:
                    await process_video_file(input_source, use_local_models)
                
                logger.info(f"Finished experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}")
                
                # Save intermediate results after each model combination
                save_current_state()
        
        experiment_completed = True
        logger.info("All experiments completed. Saving final results...")
        save_current_state()
        
        summary = await summarize_text(conversation.original_text)
        if summary:
            print(f"Summary: {summary}")
            with open(os.path.join(OUTPUT_DIR, "summary", "conversation_summary.txt"), "w") as f:
                f.write(summary)
        
        logger.info("Experiment run completed.")
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        save_current_state()
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