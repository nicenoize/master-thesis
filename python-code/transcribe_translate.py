import os
from openai import AsyncOpenAI
import subprocess
import logging
import io
import asyncio
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import make_chunks

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

CHUNK_SIZE = 16000 * 5 * 2  # 5 seconds of audio at 16kHz, 16-bit
TARGET_LANGUAGES = ['ger', 'fra', 'spa']  # German, French, Spanish
OUTPUT_DIR = "output"

class Conversation:
    def __init__(self):
        self.original_text = ""
        self.translations = {lang: "" for lang in TARGET_LANGUAGES}

    def add_text(self, text):
        self.original_text += text + "\n\n"

    def add_translation(self, lang, text):
        self.translations[lang] += text + "\n\n"

    def save_to_files(self):
        # Create output directories
        os.makedirs(os.path.join(OUTPUT_DIR, "transcription"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "summary"), exist_ok=True)
        for lang in TARGET_LANGUAGES:
            os.makedirs(os.path.join(OUTPUT_DIR, "translations", lang), exist_ok=True)

        # Save original transcription
        with open(os.path.join(OUTPUT_DIR, "transcription", "original_conversation.txt"), "w") as f:
            f.write(self.original_text)

        # Save translations
        for lang, text in self.translations.items():
            with open(os.path.join(OUTPUT_DIR, "translations", lang, f"translated_conversation_{lang}.txt"), "w") as f:
                f.write(text)

conversation = Conversation()

MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB, just under OpenAI's 26 MB limit

async def transcribe_audio(audio_chunk):
    logging.info("Starting transcription.")
    try:
        with io.BytesIO() as audio_file:
            audio_chunk.export(audio_file, format="mp3")
            audio_file.seek(0)
            response = await aclient.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.mp3", audio_file),
                response_format="text"
            )
        transcribed_text = response
        logging.info("Transcription completed.")
        return transcribed_text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

async def translate_text(text, target_lang):
    logging.info(f"Starting translation to {target_lang}.")
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}. Maintain the speaker labels if present."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        translated_text = response.choices[0].message.content.strip()
        logging.info("Translation completed.")
        return translated_text
    except Exception as e:
        logging.error(f"Error during translation to {target_lang}: {e}")
        return None

async def summarize_text(text):
    logging.info("Starting summarization.")
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize the following text concisely."},
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )
        summary = response.choices[0].message.content.strip()
        logging.info("Summarization completed.")
        return summary
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return None

async def identify_speakers(transcription):
    logging.info("Identifying speakers.")
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Identify different speakers in the following transcription. Label each line with 'Speaker 1:', 'Speaker 2:', etc. If you can't determine the speaker, use 'Unknown Speaker:'."},
                {"role": "user", "content": transcription}
            ],
            max_tokens=2000
        )
        identified_text = response.choices[0].message.content.strip()
        logging.info("Speaker identification completed.")
        return identified_text
    except Exception as e:
        logging.error(f"Error during speaker identification: {e}")
        return transcription

async def capture_and_process_stream(stream_url):
    logging.info("Starting ffmpeg process to capture audio.")
    process = await asyncio.create_subprocess_exec(
        'ffmpeg', '-i', stream_url, '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    audio_buffer = b""
    while True:
        try:
            chunk = await process.stdout.read(1024)
            if not chunk:
                logging.warning("No data read from ffmpeg process.")
                break

            audio_buffer += chunk
            if len(audio_buffer) > CHUNK_SIZE:
                logging.info("Processing 5 seconds of audio data.")
                audio_chunk = AudioSegment(
                    data=audio_buffer[:CHUNK_SIZE],
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                await process_audio_chunk(audio_chunk)
                audio_buffer = audio_buffer[CHUNK_SIZE:]  # Keep any remaining audio

        except asyncio.CancelledError:
            logging.info("Task was cancelled.")
            break
        except Exception as e:
            logging.error(f"Error while processing audio: {e}")
            break

    # Ensure the ffmpeg process is terminated
    process.terminate()
    await process.wait()


async def process_audio_chunk(audio_chunk):
    transcribed_text = await transcribe_audio(audio_chunk)

    if transcribed_text:
        identified_text = await identify_speakers(transcribed_text)
        print(f"Original with speakers: {identified_text}")
        conversation.add_text(identified_text)

        # Translate to all target languages
        translation_tasks = [translate_text(identified_text, lang) for lang in TARGET_LANGUAGES]
        translations = await asyncio.gather(*translation_tasks)

        for lang, translation in zip(TARGET_LANGUAGES, translations):
            if translation:
                print(f"Translated ({lang}): {translation}")
                conversation.add_translation(lang, translation)

async def process_video_file(file_path):
    logging.info(f"Processing video file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    
    # Calculate chunk size in milliseconds
    chunk_duration = (MAX_CHUNK_SIZE / len(audio.raw_data)) * len(audio)
    chunks = make_chunks(audio, chunk_duration)

    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i+1} of {len(chunks)}")
        await process_audio_chunk(chunk)

async def main():
    print("Choose an option:")
    print("1. Process a live stream")
    print("2. Process a video file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        stream_url = input("Enter the stream URL: ")
        await capture_and_process_stream(stream_url)
    elif choice == "2":
        file_path = input("Enter the path to the video file: ")
        await process_video_file(file_path)
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    # Save conversation to files
    conversation.save_to_files()

    # Generate and save summary
    summary = await summarize_text(conversation.original_text)
    if summary:
        print(f"Summary: {summary}")
        with open(os.path.join(OUTPUT_DIR, "summary", "conversation_summary.txt"), "w") as f:
            f.write(summary)

if __name__ == "__main__":
    asyncio.run(main())