# MLXGateway

OpenAI-compatible API server for MLX framework. All-in-one package combining mlx-lm, mlx-vlm, mlx-audio, and mflux.

## Features

- **Chat Completions** - Text generation with streaming support via mlx-lm
- **Vision** - Image understanding and analysis via mlx-vlm
- **Text-to-Speech** - Audio generation from text via mlx-audio
- **Speech-to-Text** - Transcription and translation via mlx-audio
- **Image Generation** - Text-to-image using mflux
- **Model Management** - Model caching with configurable TTL and size limits
- **Tool and Prompt Cache Support** - Using mlx-lm's tool calling and prompt cache support.
- **OpenAI Compatible** - OpenAI API endpoints

## Installation
clone this project

```bash
python3 -m venv ~/mlxgateway
source ~/mlxgateway/bin/activate

pip install -e .
```

## Usage

download model with huggingface cli 
```bash
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

```bash
mlxgateway
```

Server starts at `http://localhost:8008`

### Options

```bash
mlxgateway --host 0.0.0.0 --port 8000 --log-level info --max-models 4 --model-cache-ttl 600
```

## Thanks to

- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Language models
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Audio models (TTS & STT)
- [mflux](https://github.com/filipstrand/mflux) - Image generation
