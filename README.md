# Octy Transcribe

An advanced audio/video transcription and speaker diarization system with multiple interfaces: Telegram bot, desktop GUI, real-time microphone transcription, and a web dashboard.

Uses **OpenAI Whisper** for transcription and **Pyannote** for speaker diarization.

## Features

- **Telegram Bot** — Send audio/video files via Telegram and receive transcriptions
- **Desktop GUI** — Batch-process local media files with a tkinter interface
- **Real-time Transcription** — Live microphone transcription with streaming Whisper
- **Web Dashboard** — Real-time monitoring of bot status and processing queue (React + WebSocket)
- **Speaker Diarization** — Identify and label different speakers in recordings
- **Format Conversion** — WebM → MP3 batch conversion

## Scripts

| Script | Purpose |
|--------|---------|
| `telegram_bot.py` | Main Telegram bot with async queue, progress GUI, and WebSocket API |
| `transcribe_gui.py` | Desktop GUI for batch transcription of local media |
| `transcribe_mp3.py` | Transcribe MP3 files using OpenAI API |
| `realtime_transcribe.py` | Real-time microphone transcription |
| `diarization.py` | Speaker diarization using Pyannote |
| `merge-transcript-diarization.py` | Merge diarization output with SRT subtitles |
| `webm_to_mp3.py` | Batch WebM → MP3 converter |
| `test_mics.py` | Scan and list active microphone devices |

## Requirements

- Python 3.x
- ffmpeg & ffprobe
- CUDA Toolkit (optional, for GPU acceleration)
- Docker (optional, for local Telegram Bot API server)

### Python Dependencies

```bash
pip install openai python-telegram-bot pyannote.audio sounddevice numpy scipy whisper psutil aiohttp
```

### Frontend (Web Dashboard)

```bash
cd frontend
npm install
```

## Configuration

### 1. Telegram Bot

Copy `telegram_bot_config.example.json` to `telegram_bot_config.json` and fill in:

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "whisper_model": "base",
  "language": "it",
  "task": "transcribe",
  "fp16": false,
  "allowed_user_ids": [123456789],
  "max_file_size_mb": 2000,
  "default_output_format": "message",
  "huggingface_token": "YOUR_HF_TOKEN",
  "diarization_model": "pyannote/speaker-diarization-3.1",
  "local_api_server": {
    "enabled": true,
    "api_url": "http://localhost:8081/bot{token}/{method}",
    "file_url": "http://localhost:8081/file/bot{token}/{path}"
  }
}
```

### 2. Docker Telegram API (optional)

Create `.env.telegram-api` with your Telegram API credentials (from https://my.telegram.org/apps):

```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

Then start with:

```bash
docker compose -f docker-compose.telegram-api.yml up -d
```

## Usage

### Telegram Bot

```bash
python telegram_bot.py
# Or: start_bot.bat
```

Starts the bot with polling, opens a progress GUI, and serves the web API on `http://localhost:3001`.

### Desktop GUI

```bash
python transcribe_gui.py
# Or: start.bat
```

### Real-time Microphone Transcription

```bash
python test_mics.py                # Find your microphone device
python realtime_transcribe.py --model base --language it --chunk 5
```

### Batch Convert WebM to MP3

```bash
python webm_to_mp3.py
```

### Transcribe MP3 Files (OpenAI API)

```bash
set OPENAI_API_KEY=sk-...
python transcribe_mp3.py
```

## Project Structure

```
├── telegram_bot.py              # Main bot server
├── transcribe_gui.py            # Desktop GUI
├── transcribe_mp3.py            # OpenAI API transcriber
├── realtime_transcribe.py       # Microphone transcriber
├── diarization.py               # Speaker diarization
├── merge-transcript-diarization.py
├── webm_to_mp3.py               # Format converter
├── test_mics.py                 # Mic scanner
├── frontend/                    # React web dashboard
│   └── src/
├── input/                       # Input media files
├── mp3/                         # Converted MP3s
├── transcripts/                 # Transcription output
├── diarization/                 # Diarization output
└── Video/                       # Video files
```
