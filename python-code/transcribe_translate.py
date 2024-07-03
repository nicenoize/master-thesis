import openai
import subprocess
import wave
import os
import logging
import io
import asyncio
import aiohttp
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Load OpenAI API key from environment variable for security
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

CHUNK_SIZE = 16000 * 5 * 2  # 5 seconds of audio at 16kHz, 16-bit
TARGET_LANGUAGES = ['ger', 'fra', 'spa']  # German, French, Spanish

async def transcribe_audio(audio_data):
    logging.info("Starting transcription.")
    try:
        response = await openai.Audio.atranscribe(
            model="whisper-1",
            file=io.BytesIO(audio_data),
            response_format="json"
        )
        transcribed_text = response['text']
        logging.info("Transcription completed.")
        return transcribed_text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

async def translate_text(text, target_lang):
    logging.info(f"Starting translation to {target_lang}.")
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        translated_text = response['choices'][0]['message']['content'].strip()
        logging.info("Translation completed.")
        return translated_text
    except Exception as e:
        logging.error(f"Error during translation to {target_lang}: {e}")
        return None

async def capture_and_process_audio(stream_url):
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

                # Transcribe using Whisper
                transcribed_text = await transcribe_audio(audio_buffer[:CHUNK_SIZE])

                if transcribed_text:
                    print(f"Original: {transcribed_text}")

                    # Translate to all target languages
                    translation_tasks = [translate_text(transcribed_text, lang) for lang in TARGET_LANGUAGES]
                    translations = await asyncio.gather(*translation_tasks)

                    for lang, translation in zip(TARGET_LANGUAGES, translations):
                        if translation:
                            print(f"Translated ({lang}): {translation}")

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

async def main():
    stream_url = 'https://bintu-play.nanocosmos.de/h5live/http/stream.mp4?url=rtmp://localhost/play&stream=sNVi5-QbbRi'
    await capture_and_process_audio(stream_url)

if __name__ == "__main__":
    asyncio.run(main())