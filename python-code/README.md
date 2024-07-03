# Voice Isolation, Transcription, and Translation Tool

This repository contains tools for voice isolation, transcription, and translation from video or audio sources. It includes two main components: a batch processor for video files and a livestream translator.

## Components

1. `voiceIsolation_translation_transcription.py`: Processes video files, isolating individual speakers, transcribing their speech, and translating the transcriptions.
2. `transcribe_translate.py`: A real-time audio transcription and translation tool for livestreams or microphone input.

## Requirements

The project dependencies are listed in `requirements.txt`. To install them, run: 'pip install -r requirements.txt'

Note: This project requires Python 3.8 or higher.

## Usage

### Voice Isolation, Transcription, and Translation from Video

To process a video file, isolate speakers, transcribe their speech, and translate the transcriptions: 'python voiceIsolation_translation_transcription.py /path/to/your/video.mp4 --languages de fr --use_gpu'

Arguments:
- `video_path`: Path to the input video file (required)
- `--languages`: Target languages for translation (default: de, it)
- `--use_gpu`: Use GPU for processing if available (optional)

### Livestream Translation

To start the real-time audio transcription and translation: 'python transcribe_translate.py'

This will capture audio from your default microphone, transcribe it, and provide translations in real-time.

## Configuration

- Set up a `.env` file in the root directory with your Hugging Face token: 'HUGGING_FACE_TOKEN=your_token_here'

## Notes

- The voice isolation and transcription process can be memory-intensive. If you encounter memory issues, try processing shorter video clips or adjusting the chunk size in the code.
- GPU acceleration is supported but may not work on all systems. If you encounter issues, try running without the `--use_gpu` flag.
- The livestream translator currently supports English as the source language and German and Italian as target languages by default. You can modify these in the `LivestreamTranslator` class initialization.

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed.
2. Check that your Hugging Face token is correctly set in the `.env` file.
3. Try running the scripts without GPU acceleration.
4. For memory issues, try processing shorter audio segments or adjusting the chunk size in the code.

## License

tba
