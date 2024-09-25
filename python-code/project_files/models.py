import asyncio
from asyncio import Semaphore, Queue
from functools import partial
import aiohttp
import io
import time
import random
import numpy as np
import librosa
import torch
import os
import json
import requests
import config
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)
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

# Define rate limits for different models and APIs
RATE_LIMITS = {
    'gpt-3.5-turbo': {'tpm': 90000, 'rpm': 3500, 'max_tokens': 4096},
    'gpt-4': {'tpm': 40000, 'rpm': 200, 'max_tokens': 8192},
    'whisper-1': {'rpm': 50},
    # Add other API rate limits here
}

concurrency_limit = 3  # Adjust this value based on your needs
concurrency_semaphore = Semaphore(concurrency_limit)

api_request_queue = Queue()

# Create a dictionary to store limiters for each model/API
limiters = {}

for model, limits in RATE_LIMITS.items():
    tpm_limit = limits.get('tpm')
    rpm_limit = limits.get('rpm')
    
    if tpm_limit:
        limiters[f"{model}_tpm"] = AsyncLimiter(max_rate=tpm_limit, time_period=60)
        logger.info(f"Set TPM limiter for {model}: {tpm_limit} tokens per minute")
    if rpm_limit:
        limiters[f"{model}_rpm"] = AsyncLimiter(max_rate=rpm_limit, time_period=60)
        logger.info(f"Set RPM limiter for {model}: {rpm_limit} requests per minute")

async def process_api_request_queue(loop):
    while True:
        request_func = await api_request_queue.get()
        try:
            result = await request_func()
            api_request_queue.task_done()
            return result
        except Exception as e:
            logger.error(f"Error processing queued request: {e}")
            api_request_queue.task_done()
        await asyncio.sleep(1 + random.random())  # Add jitter

async def log_rate_limit_usage():
    for model, limits in RATE_LIMITS.items():
        tpm_limiter = limiters.get(f"{model}_tpm")
        rpm_limiter = limiters.get(f"{model}_rpm")
        
        if tpm_limiter:
            logger.info(f"{model} TPM usage: {tpm_limiter.current_rate:.2f}/{limits['tpm']} tokens per minute")
        if rpm_limiter:
            logger.info(f"{model} RPM usage: {rpm_limiter.current_rate:.2f}/{limits['rpm']} requests per minute")

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    "sentiment-analysis", device="cuda" if torch.cuda.is_available() else "cpu"
)

emotion_detector = FER(mtcnn=True)

async def get_token_count(text, model):
    return num_tokens_from_string(text, model)

async def split_text(text, model, max_tokens):
    tokens = await get_token_count(text, model)
    if tokens <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in text.split(". "):
        sentence_tokens = await get_token_count(sentence, model)
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + ". "
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def make_api_request(url, method='POST', headers=None, data=None, files=None, api_key=None, model=None, max_tokens=None, loop=None):
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set the appropriate environment variable.")

    headers = headers or {}
    headers['Authorization'] = f'Bearer {api_key}'

    tpm_limiter = limiters.get(f"{model}_tpm")
    rpm_limiter = limiters.get(f"{model}_rpm")

    logger.info(f"Queueing API request to {url} for model {model}")

    async def execute_request():
        try:
            async with concurrency_semaphore:
                if tpm_limiter:
                    await tpm_limiter.acquire()
                if rpm_limiter:
                    await rpm_limiter.acquire()

                async with aiohttp.ClientSession(loop=loop) as session:
                    if files:
                        form_data = aiohttp.FormData()
                        if data:
                            for key, value in data.items():
                                form_data.add_field(key, str(value))
                        for key, value in files.items():
                            form_data.add_field(name=key, value=value[1], filename=value[0], content_type=value[2])
                        data_to_send = form_data
                    else:
                        if data:
                            headers['Content-Type'] = 'application/json'
                            data_to_send = json.dumps(data)
                        else:
                            data_to_send = None

                    start_time = time.time()
                    async with session.request(method, url, headers=headers, data=data_to_send, timeout=30) as response:
                        elapsed_time = time.time() - start_time
                        logger.info(f"API request completed in {elapsed_time:.2f} seconds with status {response.status}")
                        
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', '1'))
                            logger.warning(f"Rate limit hit for {model}. Server requested retry after {retry_after} seconds.")
                            logger.warning(f"Response headers: {response.headers}")
                            await asyncio.sleep(retry_after + random.random())  # Add jitter
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=response.reason,
                                headers=response.headers,
                            )
                        response.raise_for_status()
                        return await response.json()
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for {model}. Retrying...")
            raise
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                logger.warning(f"Rate limit exceeded for {model}. Retrying with backoff...")
                raise
            else:
                logger.error(f"Unexpected error for {model}: {e}")
                raise

    request_func = partial(execute_request)
    await api_request_queue.put(request_func)
    return await process_api_request_queue(loop)

async def process_large_request(url, headers, data, model, max_tokens, loop):
    full_response = ""
    text = data['messages'][-1]['content']
    chunks = await split_text(text, model, max_tokens)

    for chunk in chunks:
        chunk_data = data.copy()
        chunk_data['messages'][-1]['content'] = chunk
        result = await make_api_request(url, headers=headers, data=chunk_data, model=model, max_tokens=max_tokens, loop=loop)
        full_response += result['choices'][0]['message']['content'].strip() + " "

    return full_response.strip()

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

async def detailed_analysis(transcription, audio_features, video_emotions, use_local_models=False, loop=None):
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

            # Construct the analysis prompt
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

            # Prepare data for API request
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
            
            try:
                result = await make_api_request(url, headers=headers, data=data, model=config.CURRENT_GPT_MODEL, max_tokens=2000, loop=loop)
                analysis_result = result['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.error(f"Error during API request for detailed analysis: {e}")
                analysis_result = f"Error in analysis: {str(e)}"

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

async def transcribe_audio(audio_chunk, use_local_model=False, loop=None):
    logger.info(f"Starting transcription. Use local model: {use_local_model}")
    start_time = time.time()
    duration_seconds = len(audio_chunk) / 1000  # len(audio_chunk) is in milliseconds
    if duration_seconds < 0.1:
        logger.warning("Audio chunk too short; skipping transcription.")
        return None
    try:
        if use_local_model:
            # Local model code remains unchanged
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
            try:
                result = await make_api_request(url, data=data, files=files, model='whisper-1', loop=loop)
                transcription = result.get("text", "")
                logger.info(f"Transcription result: {transcription[:100]}...")
            except Exception as e:
                logger.error(f"Error during API request for transcription: {e}")
                return None

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

async def translate_text(text_or_generator, target_lang, use_local_model=False, loop=None):
    logger.info(f"Starting translation to {target_lang}.")
    start_time = time.time()
    try:
        if use_local_model:
            # Local model code remains unchanged
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
            try:
                result = await make_api_request(url, headers=headers, data=data, model=config.CURRENT_GPT_MODEL, max_tokens=1000, loop=loop)
                translation = result['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.error(f"Error during API request for translation: {e}")
                return None

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

async def summarize_text(text, use_local_models=False, loop=None):
    logger.info("Starting summarization.")
    try:
        if use_local_models:
            # Local model code remains unchanged
            pass
        else:
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "model": config.CURRENT_GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely.",
                    },
                    {"role": "user", "content": text},
                ],
                "max_tokens": 500,
            }
            url = 'https://api.openai.com/v1/chat/completions'
            try:
                summary = await process_large_request(url, headers, data, config.CURRENT_GPT_MODEL, 4000, loop)
            except Exception as e:
                logger.error(f"Error during API request for summarization: {e}")
                return None

        logger.info("Summarization completed.")
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
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
            async with aiohttp.ClientSession() as session:
                async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                line = line[len('data: '):]
                            if line == '[DONE]':
                                break
                            try:
                                chunk = json.loads(line)
                                delta = chunk['choices'][0]['delta']
                                content = delta.get('content', '')
                                print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                logger.error(f"Error decoding JSON: {line}")
            print()  # Newline after each segment
    except Exception as e:
        logger.error(f"Error during streaming translation: {e}", exc_info=True)

async def make_custom_api_request(url, method='POST', headers=None, data=None, files=None, api_key=None, rate_limit=None):
    if rate_limit:
        custom_limiter = AsyncLimiter(max_rate=rate_limit['rpm'], time_period=60)
        async with custom_limiter:
            return await make_api_request(url, method, headers, data, files, api_key)
    else:
        return await make_api_request(url, method, headers, data, files, api_key)

# Example usage of the new custom API request function
async def call_custom_api(endpoint, data, api_key, rate_limit):
    url = f"https://api.example.com/{endpoint}"
    headers = {
        "Content-Type": "application/json",
    }
    return await make_custom_api_request(url, headers=headers, data=data, api_key=api_key, rate_limit=rate_limit)
