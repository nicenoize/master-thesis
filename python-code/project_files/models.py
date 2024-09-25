import asyncio
import aiohttp
import io
import time
import numpy as np
import librosa
import torch
import os
import json
import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiolimiter import AsyncLimiter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from utils import num_tokens_from_string
from fer import FER
import tiktoken  # For token counting
import whisper
from TTS.utils.synthesizer import Synthesizer

logger = config.logger

# Initialize local transcription model
local_whisper_model = None  # Will be loaded when needed

# Initialize local translation models
local_translation_tokenizer = {}
local_translation_model = {}

# Initialize local TTS
local_tts_synthesizer = None

# Define rate limits per model
model_rate_limits = {
    'gpt-4': {'rpm': 500, 'tpm': 40000, 'max_context_length': 8192},
    'gpt-4-turbo': {'rpm': 500, 'tpm': 90000, 'max_context_length': 8192},
    'gpt-3.5-turbo': {'rpm': 3500, 'tpm': 2000000, 'max_context_length': 4096},
    'whisper-1': {'rpm': 50},
    'tts-1': {'rpm': 50},
    'tts-1-hd': {'rpm': 3},
    # Add other models if necessary
}

# Create rate limiters per model
rate_limiters = {}
for model_name, limits in model_rate_limits.items():
    rate_limiters[model_name] = AsyncLimiter(max_rate=limits['rpm'], time_period=60)

# Token bucket implementation for TPM
class TokenBucket:
    def __init__(self, tokens_per_minute):
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.fill_rate = tokens_per_minute / 60  # tokens per second
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            if tokens <= self.tokens:
                self.tokens -= tokens
                return
            else:
                needed_tokens = tokens - self.tokens
                wait_time = needed_tokens / self.fill_rate
                logger.info(f"TokenBucket: Waiting for {wait_time:.2f} seconds due to TPM limit.")
                await asyncio.sleep(wait_time)
                self.tokens = 0

# Create token buckets per model
token_buckets = {}
for model_name, limits in model_rate_limits.items():
    tpm = limits.get('tpm')
    if tpm:
        token_buckets[model_name] = TokenBucket(tokens_per_minute=tpm)

# Ensure that your OpenAI API key is set
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Initialize sentiment analyzer
sentiment_analyzer = pipeline(
    "sentiment-analysis", device=0 if torch.cuda.is_available() else -1
)

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Function to count tokens in messages
def num_tokens_from_messages(messages, model):
    encoding = tiktoken.encoding_for_model(model)
    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1  # If there's a name, the role is omitted
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"Token counting not implemented for model {model}")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Assistant's reply priming
    return num_tokens

async def make_openai_request(
    url, method='POST', headers=None, data=None, files=None, model_name=None, expect_binary=False
):
    headers = headers or {}
    headers['Authorization'] = f'Bearer {openai_api_key}'

    rpm_limiter = rate_limiters.get(model_name, AsyncLimiter(1, 60))
    tpm_bucket = token_buckets.get(model_name)

    # Estimate tokens to be used
    total_tokens = 0
    if data and 'messages' in data and 'model' in data:
        # It's a chat completion request
        model = data['model']
        messages = data['messages']
        max_tokens = data.get('max_tokens', 0)
        prompt_tokens = num_tokens_from_messages(messages, model)
        total_tokens = prompt_tokens + max_tokens
    elif data and 'input' in data:
        # For other endpoints like embeddings
        model = data['model']
        input_text = data['input']
        encoding = tiktoken.encoding_for_model(model)
        total_tokens = len(encoding.encode(input_text))
    # You can add more cases as needed

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientResponseError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def call_api():
        if tpm_bucket and total_tokens > 0:
            await tpm_bucket.consume(total_tokens)
        async with rpm_limiter:
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
                else:
                    # For non-file requests, send data as JSON
                    if data:
                        headers['Content-Type'] = 'application/json'
                        data_to_send = json.dumps(data)
                    else:
                        data_to_send = None

                async with session.request(method, url, headers=headers, data=data_to_send) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                    if response.status == 429 or response.status == 400:
                        error_data = await response.json()
                        error_code = error_data.get('error', {}).get('code')
                        if error_code == 'rate_limit_exceeded':
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                await asyncio.sleep(float(retry_after))
                            else:
                                await asyncio.sleep(60)  # Default wait time
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=response.reason,
                                headers=response.headers,
                            )
                    response.raise_for_status()
                    if expect_binary:
                        return await response.read()
                    else:
                        return await response.json()
    return await call_api()

def split_text_into_chunks(text, max_tokens, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    return chunks

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
            sentiment_result = sentiment_analyzer(transcription)[0]
            sentiment = sentiment_result['label']
            analysis_result = f"Transcription: {transcription}\n"
            analysis_result += f"Sentiment: {sentiment} (score: {sentiment_result['score']:.2f})\n"
            analysis_result += "Audio Features: MFCCs and Chroma data available\n" if audio_features else "Audio Features: Not available\n"
            analysis_result += f"Video Emotions: {video_emotions}\n" if video_emotions else "Video Emotions: Not available\n"
        else:
            model_name = config.CURRENT_GPT_MODEL
            limits = model_rate_limits.get(model_name, {})
            max_context_length = limits.get('max_context_length', 8192)
            max_response_tokens = 1000  # Adjust as needed
            encoding = tiktoken.encoding_for_model(model_name)

            # Prepare audio features and video emotions for the prompt
            audio_features_str = "Audio features available." if audio_features else "No audio features extracted."
            video_emotions_str = "Video emotions available." if video_emotions else "No video emotions extracted."

            # Build the analysis prompt
            base_prompt = f"""
Analyze the following transcription, taking into account the provided audio features and video emotions:

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

            base_prompt_tokens = len(encoding.encode(base_prompt))
            max_transcription_tokens = max_context_length - base_prompt_tokens - max_response_tokens - 100  # Some buffer

            # Split transcription into chunks
            transcription_chunks = split_text_into_chunks(transcription, max_transcription_tokens, model_name)

            analysis_results = []

            for chunk in transcription_chunks:
                analysis_prompt = base_prompt + f"\nTranscription: {chunk}\n"

                headers = {
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert in multimodal sentiment analysis, capable of interpreting text, audio features, and visual emotional cues.",
                        },
                        {"role": "user", "content": analysis_prompt},
                    ],
                    "max_tokens": max_response_tokens,
                }
                url = 'https://api.openai.com/v1/chat/completions'

                # Estimate tokens
                prompt_tokens = num_tokens_from_messages(data['messages'], model_name)
                total_tokens = prompt_tokens + data['max_tokens']

                # Make the API request
                result = await make_openai_request(url, headers=headers, data=data, model_name=model_name)
                analysis_result = result['choices'][0]['message']['content'].strip()
                analysis_results.append(analysis_result)

            # Combine analysis results
            final_analysis = "\n".join(analysis_results)

            # Perform sentiment analysis on the transcription or analysis_result
            sentiment_result = sentiment_analyzer(transcription)[0]
            sentiment = sentiment_result['label']

        total_time = time.time() - start_time
        logger.info(f"Analysis completed in {total_time:.2f} seconds.")
        return {'analysis_result': final_analysis, 'sentiment': sentiment}
    except Exception as e:
        logger.error(f"Error during detailed analysis: {e}", exc_info=True)
        return {'analysis_result': transcription, 'sentiment': 'Neutral'}

async def transcribe_audio(audio_chunk, use_local_model=False):
    logger.info(f"Starting transcription. Use local model: {use_local_model}")
    start_time = time.time()
    duration_seconds = len(audio_chunk) / 1000  # len(audio_chunk) is in milliseconds
    if duration_seconds < 0.1:
        logger.warning("Audio chunk too short; skipping transcription.")
        return None
    try:
        if use_local_model:
            global local_whisper_model
            if local_whisper_model is None:
                # Load the Whisper model
                logger.info("Loading local Whisper model...")
                local_whisper_model = whisper.load_model('base')  # Choose model size as needed
            # Convert audio_chunk to numpy array
            audio_data = np.array(audio_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
            # Resample if necessary
            sample_rate = audio_chunk.frame_rate
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            # Transcribe using local model
            result = local_whisper_model.transcribe(audio_data, language='en')
            transcription = result['text']
            logger.info(f"Transcription result: {transcription[:100]}...")
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
            result = await make_openai_request(url, data=data, files=files, model_name='whisper-1')
            transcription = result.get("text", "")
            logger.info(f"Transcription result: {transcription[:100]}...")
        total_time = time.time() - start_time
        logger.info(f"Transcription completed in {total_time:.2f} seconds.")
        return transcription if transcription else None
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        logger.error(
            f"Audio chunk details: Duration: {len(audio_chunk) / 1000}s, Frame rate: {audio_chunk.frame_rate}, Channels: {audio_chunk.channels}"
        )
        return None

async def translate_text(text_or_generator, target_lang, sentiment=None, use_local_model=False):
    logger.info(f"Starting translation to {target_lang}. Use local model: {use_local_model}")
    start_time = time.time()
    translation = None  # Initialize translation variable
    try:
        if use_local_model:
            global local_translation_tokenizer, local_translation_model
            model_key = f'en-{target_lang}'
            if model_key not in local_translation_model:
                # Load the local translation model
                logger.info(f"Loading local translation model for {model_key}...")
                model_name = f'Helsinki-NLP/opus-mt-en-{target_lang}'
                local_translation_tokenizer[model_key] = AutoTokenizer.from_pretrained(model_name)
                local_translation_model[model_key] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = local_translation_tokenizer[model_key]
            model = local_translation_model[model_key]
            # Translate the text
            inputs = tokenizer(text_or_generator, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Translation result: {translation[:100]}...")
        elif isinstance(text_or_generator, str):
            model_name = config.CURRENT_GPT_MODEL
            headers = {
                "Content-Type": "application/json",
            }
            data = {
                "model": model_name,
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
            result = await make_openai_request(url, headers=headers, data=data, model_name=model_name)
            translation = result['choices'][0]['message']['content'].strip()
            logger.info(f"Translation result: {translation[:100]}...")
        else:
            # Handle streaming translation if needed
            translation = None
        total_time = time.time() - start_time
        logger.info(f"Translation completed in {total_time:.2f} seconds.")
        # After translation, generate speech
        if translation and sentiment:
            speech_result = await synthesize_speech(translation, target_lang, sentiment, use_local_model=use_local_model)
            # Handle the speech result (e.g., save to file, play audio)
        return translation
    except Exception as e:
        logger.error(f"Error during translation: {e}", exc_info=True)
        return None

async def summarize_text(text, use_local_models=False):
    logger.info("Starting summarization.")
    try:
        model_name = config.CURRENT_GPT_MODEL
        limits = model_rate_limits.get(model_name, {})
        max_context_length = limits.get('max_context_length', 8192)
        max_response_tokens = 500  # Adjust as needed
        encoding = tiktoken.encoding_for_model(model_name)

        if use_local_models:
            # Use local summarization model
            summarizer = pipeline("summarization")
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        else:
            # Split text into chunks
            base_prompt = "Summarize the following text concisely:"
            base_prompt_tokens = len(encoding.encode(base_prompt))
            max_text_tokens = max_context_length - base_prompt_tokens - max_response_tokens - 100  # Some buffer

            text_chunks = split_text_into_chunks(text, max_text_tokens, model_name)
            summaries = []

            for chunk in text_chunks:
                prompt = base_prompt + f"\n{chunk}"

                headers = {
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_response_tokens,
                }
                url = 'https://api.openai.com/v1/chat/completions'

                # Estimate tokens
                prompt_tokens = num_tokens_from_messages(data['messages'], model_name)
                total_tokens = prompt_tokens + data['max_tokens']

                result = await make_openai_request(url, headers=headers, data=data, model_name=model_name)
                summary_chunk = result['choices'][0]['message']['content'].strip()
                summaries.append(summary_chunk)

            summary = " ".join(summaries)

        logger.info("Summarization completed.")
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        return None

async def synthesize_speech(text, language, sentiment, model='tts-1', use_local_model=False):
    logger.info(f"Starting speech synthesis. Use local model: {use_local_model}")
    if use_local_model:
        global local_tts_synthesizer
        if local_tts_synthesizer is None:
            # Initialize the local TTS synthesizer
            logger.info("Loading local TTS model...")
            # Adjust the model and vocoder paths as necessary
            tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            vocoder_name = "vocoder_models/en/ljspeech/hifigan_v2"
            local_tts_synthesizer = Synthesizer(tts_model_name, vocoder_name)
        # Generate speech
        wav = local_tts_synthesizer.tts(text)
        # Save to file
        audio_filename = f"output_{language}_{int(time.time())}.wav"
        local_tts_synthesizer.save_wav(wav, audio_filename)
        logger.info(f"Saved synthesized speech to {audio_filename}")
        return audio_filename
    else:
        logger.info(f"Starting speech synthesis with model {model}.")
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            'model': model,
            'text': text,
            'language': language,
            'sentiment': sentiment,
        }
        url = 'https://api.openai.com/v1/audio/synthesize'
        result = await make_openai_request(url, headers=headers, data=data, model_name=model, expect_binary=True)
        # Save or process the audio data
        audio_filename = f"output_{language}_{int(time.time())}.mp3"
        with open(audio_filename, 'wb') as f:
            f.write(result)
        logger.info(f"Saved synthesized speech to {audio_filename}")
        return audio_filename
