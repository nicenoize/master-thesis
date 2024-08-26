import asyncio
from typing import AsyncGenerator

class StreamingProcessor:
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    async def process_stream(self, stream: AsyncGenerator, processor: callable) -> AsyncGenerator:
        buffer = b""
        async for chunk in stream:
            buffer += chunk
            while len(buffer) >= self.chunk_size:
                to_process = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                result = await processor(to_process)
                yield result
        if buffer:
            result = await processor(buffer)
            yield result

    async def audio_stream(self, audio_file: str) -> AsyncGenerator:
        with open(audio_file, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
                await asyncio.sleep(0.1)  # Simulate real-time streaming

    async def process_audio(self, audio_file: str, processor: callable) -> AsyncGenerator:
        audio_stream = self.audio_stream(audio_file)
        async for result in self.process_stream(audio_stream, processor):
            yield result