# processing.py

import asyncio
import os
import cv2
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import time

import config

from models import (
    transcribe_audio,
    analyze_audio_features,
    analyze_video_frame,
    detailed_analysis,
    translate_text,
    summarize_text,
)
from classes import Conversation
from utils import validate_file_path, save_current_state

conversation = Conversation()
logger = config.logger

async def process_chunk(audio_chunk, video_frame=None, use_local_models=False):
    start_time = time.time()
    logger.info("Processing a new chunk.")
    transcribed_text = await transcribe_audio(audio_chunk, use_local_models)
    if transcribed_text:
        logger.info(f"Transcribed text: {transcribed_text[:100]}...")
        audio_features = await analyze_audio_features(audio_chunk)
        logger.info(f"Audio features extracted: {audio_features}")
        video_emotions = (
            await analyze_video_frame(video_frame) if video_frame is not None else None
        )
        logger.info(f"Video emotions: {video_emotions}")
        detailed_analysis_result = await detailed_analysis(
            transcribed_text, audio_features, video_emotions, use_local_models
        )
        logger.info(f"Detailed analysis: {detailed_analysis_result}")
        conversation.add_text(detailed_analysis_result)

        # Translate concurrently
        translation_tasks = [
            translate_text(detailed_analysis_result, lang, use_local_models)
            for lang in config.TARGET_LANGUAGES
        ]
        translations = await asyncio.gather(*translation_tasks)

        for lang, translation in zip(config.TARGET_LANGUAGES, translations):
            if translation:
                logger.info(f"Translated ({lang}): {translation}")
                conversation.add_translation(lang, translation)
    else:
        logger.warning("Transcription failed; skipping analysis and translation.")

    total_time = time.time() - start_time
    logger.info(f"Finished processing chunk in {total_time:.2f} seconds.")
    config.PERFORMANCE_LOGS["total_processing"].setdefault(
        f"{'local' if use_local_models else 'api'}_{config.CURRENT_GPT_MODEL}_{config.CURRENT_WHISPER_MODEL}",
        [],
    ).append(total_time)

async def process_video_file(file_path, use_local_models=False):
    logger.info(f"Processing video file: {file_path}")
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info(
        f"Video properties: FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f} seconds"
    )

    audio = AudioSegment.from_file(file_path)
    chunk_duration = 5000  # 5 seconds in milliseconds
    audio_chunks = make_chunks(audio, chunk_duration)

    semaphore = asyncio.Semaphore(5)  # Adjust based on your API rate limits

    async def process_chunk_wrapper(i, chunk):
        async with semaphore:
            start_time = i * chunk_duration / 1000
            end_time = min((i + 1) * chunk_duration / 1000, duration)
            timestamp = (start_time + end_time) / 2
            if timestamp > duration:
                timestamp = duration - 0.1

            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = video.read()

            if ret:
                await process_chunk(chunk, frame, use_local_models)
            else:
                logger.warning(f"Could not read frame at time {timestamp:.2f} seconds")
                await process_chunk(chunk, use_local_models=use_local_models)

    tasks = [process_chunk_wrapper(i, chunk) for i, chunk in enumerate(audio_chunks)]
    await asyncio.gather(*tasks)

    logger.info("Finished processing video file")
    video.release()

async def run_experiment(input_source, use_local_models=False):
    gpt_models = ["gpt-4", "gpt-4-0613"]
    whisper_models = ["base", "small", "medium", "large"]

    try:
        for gpt_model in gpt_models:
            for whisper_model in whisper_models:
                config.CURRENT_GPT_MODEL = gpt_model
                config.CURRENT_WHISPER_MODEL = whisper_model

                logger.info(
                    f"Starting experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}"
                )

                if isinstance(input_source, str) and input_source.startswith("rtmp://"):
                    await capture_and_process_stream(input_source, use_local_models)
                else:
                    await process_video_file(input_source, use_local_models)

                logger.info(
                    f"Finished experiment with GPT model: {gpt_model}, Whisper model: {whisper_model}"
                )

                # Save intermediate results after each model combination
                save_current_state()

        logger.info("All experiments completed. Saving final results...")
        save_current_state()

        if conversation.original_text.strip():
            summary = await summarize_text(conversation.original_text, use_local_models)
            if summary:
                logger.info(f"Summary: {summary}")
                with open(
                    os.path.join(config.OUTPUT_DIR, "summary", "conversation_summary.txt"), "w"
                ) as f:
                    f.write(summary)
        else:
            logger.warning("No text available for summarization.")

        logger.info("Experiment run completed.")
    except Exception as e:
        logger.error(f"Error during experiment: {e}", exc_info=True)
        save_current_state()
