import asyncio
from config import get_output_structure
from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.text_processor import TextProcessor
from utils.performance_logger import PerformanceLogger

async def process_livestream(stream_url, use_local_models, perform_additional_analysis, environment):
    output_structure = get_output_structure(
        environment, 
        "local" if use_local_models else "api",
        "large",  # Assuming large model for livestream
        "gpt-3.5-turbo" if not use_local_models else "local"
    )

    audio_processor = AudioProcessor()
    video_processor = VideoProcessor()
    text_processor = TextProcessor()
    performance_logger = PerformanceLogger()

    chunk_queue = asyncio.Queue()
    
    producer = asyncio.create_task(chunk_producer(stream_url, chunk_queue))
    consumers = [asyncio.create_task(chunk_consumer(chunk_queue, use_local_models, perform_additional_analysis, output_structure, performance_logger)) for _ in range(3)]
    
    await producer
    await chunk_queue.join()
    for consumer in consumers:
        consumer.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)

    performance_logger.generate_report(environment, use_local_models)

async def chunk_producer(stream_url, chunk_queue):
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
        chunk = await process.stdout.read(1024)
        if not chunk:
            break

        audio_buffer += chunk
        if len(audio_buffer) > config["CHUNK_SIZE"]:
            audio_chunk = AudioSegment(
                data=audio_buffer[:config["CHUNK_SIZE"]],
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
            audio_buffer = audio_buffer[config["CHUNK_SIZE"]:]

    process.terminate()
    await process.wait()
    await chunk_queue.put(None)  # Signal that production is done

async def chunk_consumer(chunk_queue, use_local_models, perform_additional_analysis, output_structure, performance_logger):
    audio_processor = AudioProcessor()
    video_processor = VideoProcessor()
    text_processor = TextProcessor()

    while True:
        chunk_data = await chunk_queue.get()
        if chunk_data is None:
            break

        audio_chunk, video_frame = chunk_data

        with performance_logger.measure_time("transcription"):
            transcribed_text = await audio_processor.api_transcribe(audio_chunk) if not use_local_models else await WhisperModel("large").transcribe(audio_chunk)

        with performance_logger.measure_time("translation"):
            translations = await text_processor.translate(transcribed_text, "gpt-3.5-turbo" if not use_local_models else GPTModel("local"))

        results = {
            "transcription": transcribed_text,
            "translations": translations,
        }

        if perform_additional_analysis:
            with performance_logger.measure_time("sentiment_analysis"):
                results["sentiment_analysis"] = await text_processor.analyze_sentiment(transcribed_text, "gpt-3.5-turbo" if not use_local_models else SentimentModel())
            
            with performance_logger.measure_time("video_analysis"):
                if video_frame is not None:
                    results["video_analysis"] = await video_processor.analyze_video_frame(video_frame)

        save_chunk_results(results, output_structure)
        chunk_queue.task_done()

def save_chunk_results(results, output_structure):
    for key, path in output_structure.items():
        os.makedirs(path, exist_ok=True)
        if key in results:
            with open(os.path.join(path, f"{key}_{time.time()}.json"), "w") as f:
                json.dump(results[key], f, indent=2)