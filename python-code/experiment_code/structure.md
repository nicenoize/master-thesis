```
EXPERIMENT_CODE/
│
├── api/
│   ├── __init__.py
│   ├── huggingface_api.py
│   ├── openai_api.py
│   ├── speechmatics_api.py
│   ├── elevenlabs_api.py
│   └── rateLimiter.py
│
├── evaluation_metrics/
│   ├── __init__.py
│   ├── evaluation_metrics.py
│   ├── accuracy_metrics.py
│   ├── latency_metrics.py
│   ├── scalability_metrics.py
│   └── cost_metrics.py
│
├── experiment_results/
│   └── M1 Max/
│       ├── api/
│       └── local/
│
├── experiments/
│   ├── __init__.py
│   ├── experiment.py
│   └── livestream.py
│
├── models/
│   ├── transcription/
│   │   └── whisper.py
│   ├── translation/
│   │   ├── gpt.py
│   │   └── llama.py
│   ├── sentiment/
│   │   ├── gpt.py
│   │   └── llama.py
│   └── tts/
│       └── tacotron.py
│
├── processors/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── streaming_processor.py
│   ├── text_processor.py
│   ├── video_processor.py
│   └── tts_processor.py
│
├── utils/
│   ├── __init__.py
│   ├── analysis_utils.py
│   ├── logger.py
│   ├── performance_logger.py
│   ├── plot_generator.py
│   ├── model_optimizer.py
│   └── concurrency_simulator.py
│
├── benchmarks/
│   ├── transcription_benchmark.py
│   ├── translation_benchmark.py
│   └── sentiment_benchmark.py
│
├── .env
├── config.py
├── main.py
└── structure.md
```