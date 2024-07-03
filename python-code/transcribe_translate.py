import os
import asyncio
import logging
import io
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize FER
emotion_detector = FER(mtcnn=True)

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
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception)
)
async def api_call_with_backoff(func, *args, **kwargs):
    async with rate_limit:
        return await func(*args, **kwargs)

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

async def detailed_analysis(transcription, audio_features, video_emotions):
    logger.info("Performing detailed analysis.")
    try:
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in multimodal sentiment analysis, capable of interpreting text, audio features, and visual emotional cues."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during detailed analysis: {e}")
        return transcription

async def transcribe_audio(audio_chunk):
    logger.info("Starting transcription.")
    try:
        with io.BytesIO() as audio_file:
            audio_chunk.export(audio_file, format="mp3")
            audio_file.seek(0)
            response = await api_call_with_backoff(
                aclient.audio.transcriptions.create,
                model="whisper-1",
                file=("audio.mp3", audio_file),
                response_format="text"
            )
        return response
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None

async def translate_text(text, target_lang):
    logger.info(f"Starting translation to {target_lang}.")
    try:
        response = await api_call_with_backoff(
            aclient.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}. Maintain the speaker labels if present."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during translation to {target_lang}: {e}")
        return None

async def summarize_text(text):
    logger.info("Starting summarization.")
    try:
        max_tokens = 4000
        if num_tokens_from_string(text, "gpt-4") > max_tokens:
            chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
            summaries = []
            for chunk in chunks:
                response = await api_call_with_backoff(
                    aclient.chat.completions.create,
                    model="gpt-4",
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
                model="gpt-4",
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

async def process_chunk(audio_chunk, video_frame=None):
    transcribed_text = await transcribe_audio(audio_chunk)
    if transcribed_text:
        audio_features = await analyze_audio_features(audio_chunk)
        video_emotions = await analyze_video_frame(video_frame) if video_frame is not None else None
        detailed_analysis_result = await detailed_analysis(transcribed_text, audio_features, video_emotions)
        logger.info(f"Detailed analysis: {detailed_analysis_result}")
        conversation.add_text(detailed_analysis_result)

        translation_tasks = [translate_text(detailed_analysis_result, lang) for lang in TARGET_LANGUAGES]
        translations = await asyncio.gather(*translation_tasks)

        for lang, translation in zip(TARGET_LANGUAGES, translations):
            if translation:
                logger.info(f"Translated ({lang}): {translation}")
                conversation.add_translation(lang, translation)

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

async def chunk_consumer():
    while True:
        chunk_data = await chunk_queue.get()
        if chunk_data is None:
            break
        audio_chunk, video_frame = chunk_data
        await process_chunk(audio_chunk, video_frame)
        chunk_queue.task_done()

async def capture_and_process_stream(stream_url):
    producer = asyncio.create_task(chunk_producer(stream_url))
    consumers = [asyncio.create_task(chunk_consumer()) for _ in range(5)]  # Create 5 consumers
    
    await producer
    await chunk_queue.join()
    for consumer in consumers:
        consumer.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)

async def process_video_file(file_path):
    logger.info(f"Processing video file: {file_path}")
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    video.release()

    audio = AudioSegment.from_file(file_path)
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
            await process_chunk(chunk, frame)
        else:
            await process_chunk(chunk)

    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing to 5 chunks

    async def semaphore_wrapper(i, chunk):
        async with semaphore:
            await process_chunk_wrapper(i, chunk)

    tasks = [asyncio.create_task(semaphore_wrapper(i, chunk)) for i, chunk in enumerate(chunks)]
    await asyncio.gather(*tasks)

def main():
    print("Choose an option:")
    print("1. Process a livestream")
    print("2. Process a video file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        stream_url = input("Enter the stream URL: ")
        asyncio.run(capture_and_process_stream(stream_url))
    elif choice == "2":
        file_path = input("Enter the path to the video file: ")
        asyncio.run(process_video_file(file_path))
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    conversation.save_to_files()

    summary = asyncio.run(summarize_text(conversation.original_text))
    if summary:
        print(f"Summary: {summary}")
        with open(os.path.join(OUTPUT_DIR, "summary", "conversation_summary.txt"), "w") as f:
            f.write(summary)

if __name__ == "__main__":
    main()