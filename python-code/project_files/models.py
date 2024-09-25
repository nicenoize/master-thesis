import asyncio
import aiohttp
import io
import time
import numpy as np
import librosa
import torch
import os
import json
import requests
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from utils import num_tokens_from_string
from fer import FER

logger = config.logger

# Set up rate limiters based on your actual OpenAI API rate limits
GPT_RATE_LIMIT = 60  # Replace with your actual GPT rate limit per minute
WHISPER_RATE_LIMIT = 60  # Replace with your actual Whisper rate limit per minute

gpt_limiter = AsyncLimiter(max_rate=GPT_RATE_LIMIT, time_period=60)
whisper_limiter = AsyncLimiter(max_rate=WHISPER_RATE_LIMIT, time_period=60)


# Ensure that your OpenAI API key is set
openai_api_key = os.getenv('OPENAI_API_KEY')  # Set the API key

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    "sentiment-analysis", device="cuda" if torch.cuda.is_available() else "cpu"
)

emotion_detector = FER(mtcnn=True)

async def make_openai_request(
    url, method='POST', headers=None, data=None, files=None, is_gpt=False
):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

    headers = headers or {}
    headers['Authorization'] = f'Bearer {openai_api_key}'
    limiter = gpt_limiter if is_gpt else whisper_limiter

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientResponseError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def call_api():
        async with limiter:
            async with aiohttp.ClientSession() as session:
                # Construct FormData for multipart/form-data requests
                if files:
                    form_data = aiohttp.FormData()
                    # Add data fields to form_data
                    if data:
                        for key, value in data.items():
                            form_data.add_field(key, str(value))
                    # Add files to form_data
                    for key, value in files.items():
                        form_data.add_field(
                            name=key,
                            value=value[1],  # File content
                            filename=value[0],  # Filename
                            content_type=value[2]  # Content type
                        )
                    data_to_send = form_data
                    # Do not set 'Content-Type' header; aiohttp will handle it
                else:
                    # For non-file requests, send data as JSON
                    if data:
                        headers['Content-Type'] = 'application/json'
                        data_to_send = json.dumps(data)
                    else:
                        data_to_send = None

                async with session.request(method, url, headers=headers, data=data_to_send) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            await asyncio.sleep(float(retry_after))
                        else:
                            await asyncio.sleep(1)  # Default wait time
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=response.reason,
                            headers=response.headers,
                        )
                    response.raise_for_status()
                    return await response.json()
    return await call_api()

async def analyze_audio_features(audio_chunk):
    logger.info("Analyzing audio features.")
    try:
        # Convert audio_chunk to numpy array
        audio_data = np.array(audio_chunk.get_array_of_samples())
        sample_rate = audio_chunk.frame_rate

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=sample_rate)

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data.astype(float), sr=sample_rate)

        return {
            'mfccs': mfccs.tolist(),  # Convert numpy array to list for JSON serialization
            'chroma': chroma.tolist(),
        }
    except Exception as e:
        logger.error(f"Error during audio feature extraction: {e}", exc_info=True)
        return {
            'mfccs': None,
            'chroma': None,
        }

async def analyze_video_frame(frame):
    logger.info("Analyzing video frame for emotions.")
    try:
        # Use the FER library to detect emotions
        emotions = emotion_detector.detect_emotions(frame)
        return emotions
    except Exception as e:
        logger.error(f"Error during video emotion analysis: {e}", exc_info=True)
        return None


async def detailed_analysis(transcription, audio_features, video_emotions, use_local_models=False):
    logger.info("Performing detailed analysis.")
    start_time = time.time()
    try:
        if use_local_models:
            # Perform local sentiment analysis
            sentiment = sentiment_analyzer(transcription)[0]
            analysis_result = f"Transcription: {transcription}\n"
            analysis_result += f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})\n"
            analysis_result += "Audio Features: MFCCs and Chroma data available\n" if audio_features else "Audio Features: Not available\n"
            analysis_result += f"Video Emotions: {video_emotions}\n" if video_emotions else "Video Emotions: Not available\n"
        else:
            # Prepare audio features and video emotions for the prompt
            audio_features_str = (
                f"MFCCs: {audio_features.get('mfccs', 'N/A')}\nChroma: {audio_features.get('chroma', 'N/A')}"
                if audio_features else "No audio features extracted."
            )
            video_emotions_str = (
                f"Video Emotions: {video_emotions}" if video_emotions else "No video emotions extracted."
            )

            # Use OpenAI API via make_openai_request
            analysis_prompt = f"""
Analyze the following transcription, taking into account the provided audio features and video emotions:

Transcription: {transcription}

Audio Features:
{audio_features_str}

{video_emotions_str}

Based on this information:
1. Identify the speakers.
2. Analyze the sentiment of each sentence.
3. Describe the intonation and overall vibe of each speaker's delivery.
4. Note any significant emotional changes or discrepancies between speech content and audio/visual cues.

Format your response as:
Speaker X: [Sentence] (Sentiment: [sentiment], Intonation: [description], Vibe: [description])
"""

            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "model": config.CURRENT_GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in multimodal sentiment analysis, capable of interpreting text, audio features, and visual emotional cues.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                "max_tokens": 2000,
            }
            url = 'https://api.openai.com/v1/chat/completions'
            result = await make_openai_request(url, headers=headers, data=data, is_gpt=True)
            analysis_result = result['choices'][0]['message']['content'].strip()

        total_time = time.time() - start_time
        logger.info(f"Analysis completed in {total_time:.2f} seconds.")
        config.PERFORMANCE_LOGS["analysis"].setdefault(
            f"{'local' if use_local_models else 'api'}_{config.CURRENT_GPT_MODEL}", []
        ).append(total_time)
        return analysis_result
    except Exception as e:
        logger.error(f"Error during detailed analysis: {e}", exc_info=True)
        config.PERFORMANCE_LOGS["analysis"].setdefault(
            f"{'local' if use_local_models else 'api'}_{config.CURRENT_GPT_MODEL}", []
        ).append(time.time() - start_time)
        return transcription

async def transcribe_audio(audio_chunk, use_local_model=False):
    logger.info(f"Starting transcription. Use local model: {use_local_model}")
    start_time = time.time()
    duration_seconds = len(audio_chunk) / 1000  # len(audio_chunk) is in milliseconds
    if duration_seconds < 0.1:
        logger.warning("Audio chunk too short; skipping transcription.")
        return None
    try:
        if use_local_model:
            # [Your existing code for local model]
            pass
        else:
            # Prepare audio file
            audio_file = io.BytesIO()
            audio_chunk.export(audio_file, format="mp3")
            audio_file.seek(0)
            audio_size = audio_file.getbuffer().nbytes
            logger.info(f"Audio chunk size: {audio_size} bytes")

            files = {
                'file': ('audio.mp3', audio_file.getvalue(), 'audio/mpeg'),
            }
            data = {
                'model': 'whisper-1',
            }
            url = 'https://api.openai.com/v1/audio/transcriptions'
            result = await make_openai_request(url, data=data, files=files, is_gpt=False)
            transcription = result.get("text", "")
            logger.info(f"Transcription result: {transcription[:100]}...")

        total_time = time.time() - start_time
        logger.info(f"Transcription completed in {total_time:.2f} seconds.")
        config.PERFORMANCE_LOGS["transcription"].setdefault(
            f"{'local' if use_local_model else 'api'}_{config.CURRENT_WHISPER_MODEL}",
            [],
        ).append(total_time)
        return transcription if transcription else None
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        logger.error(
            f"Audio chunk details: Duration: {len(audio_chunk) / 1000}s, Frame rate: {audio_chunk.frame_rate}, Channels: {audio_chunk.channels}"
        )
        config.PERFORMANCE_LOGS["transcription"].setdefault(
            f"{'local' if use_local_model else 'api'}_{config.CURRENT_WHISPER_MODEL}",
            [],
        ).append(time.time() - start_time)
        return None


async def translate_text(text_or_generator, target_lang, use_local_model=False):
    logger.info(f"Starting translation to {target_lang}.")
    start_time = time.time()
    try:
        if use_local_model:
            # Existing code for local translation
            pass
        elif isinstance(text_or_generator, str):
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "model": config.CURRENT_GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_lang}. Maintain the speaker labels if present.",
                    },
                    {"role": "user", "content": text_or_generator},
                ],
                "max_tokens": 1000,
            }
            url = 'https://api.openai.com/v1/chat/completions'
            result = await make_openai_request(url, headers=headers, data=data, is_gpt=True)
            translation = result['choices'][0]['message']['content'].strip()

            total_time = time.time() - start_time
            logger.info(f"Translation completed in {total_time:.2f} seconds.")
            config.PERFORMANCE_LOGS["translation"].setdefault(
                f"{'local' if use_local_model else 'api'}_{config.CURRENT_GPT_MODEL}",
                [],
            ).append(total_time)
            return translation
        else:
            # Handle streaming translation if needed
            await translate_text_streaming(text_or_generator, target_lang)
    except Exception as e:
        logger.error(f"Error during translation: {e}", exc_info=True)
        config.PERFORMANCE_LOGS["translation"].setdefault(
            f"{'local' if use_local_model else 'api'}_{config.CURRENT_GPT_MODEL}",
            [],
        ).append(time.time() - start_time)
        return None

async def translate_text_streaming(text_generator, target_lang):
    logger.info(f"Starting streaming translation to {target_lang}.")
    openai_api_key = os.getenv('OPENAI_API_KEY')
    try:
        async for text_segment in text_generator:
            logger.info("Sending translation request to OpenAI API.")
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": config.CURRENT_GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Translate the following text to {target_lang}. Maintain the speaker labels if present.",
                    },
                    {"role": "user", "content": text_segment},
                ],
                "max_tokens": 1000,
                "stream": True,
            }
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True,  # Enable streaming
            )
            response.raise_for_status()
            logger.info("Received response from OpenAI API.")

            # Process the streamed responses
            translation = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[len('data: '):]
                    if line == '[DONE]':
                        break
                    chunk = json.loads(line)
                    delta = chunk['choices'][0]['delta']
                    content = delta.get('content', '')
                    translation += content
                    # Process the partial translation here
                    print(content, end='', flush=True)
            print()  # Newline after each segment
    except Exception as e:
        logger.error(f"Error during streaming translation: {e}", exc_info=True)

async def summarize_text(text, use_local_models=False):
    logger.info("Starting summarization.")
    try:
        if use_local_models:
            # Use local summarization model
            summarizer = pipeline("summarization")
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        else:
            # Handle large texts by splitting into chunks
            max_tokens = 4000
            token_count = num_tokens_from_string(text, config.CURRENT_GPT_MODEL)
            summaries = []
            chunks = [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)] if token_count > max_tokens else [text]

            for chunk in chunks:
                headers = {
                    "Content-Type": "application/json",
                }
                data = json.dumps({
                    "model": config.CURRENT_GPT_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Summarize the following text concisely.",
                        },
                        {"role": "user", "content": chunk},
                    ],
                    "max_tokens": 500,
                })
                url = 'https://api.openai.com/v1/chat/completions'
                result = await make_openai_request(url, headers=headers, data=data, is_gpt=True)
                summary_chunk = result['choices'][0]['message']['content'].strip()
                summaries.append(summary_chunk)

            summary = " ".join(summaries)

        logger.info("Summarization completed.")
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        return None
