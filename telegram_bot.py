"""
Telegram Bot per trascrizione audio/video con Whisper.
Riceve file audio o video e restituisce la trascrizione come messaggio in chat
(aggiornato in tempo reale) oppure come file, a scelta dell'utente.

Usa polling (long-polling) quindi funziona da qualsiasi host senza IP pubblico.
Basta eseguire: python telegram_bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters,
        ContextTypes,
    )
except ImportError:
    print("Errore: python-telegram-bot non installato.")
    print("Esegui: pip install python-telegram-bot")
    sys.exit(1)

# --- Config ---
CONFIG_FILE = Path(__file__).parent / "telegram_bot_config.json"
USER_SETTINGS_FILE = Path(__file__).parent / "telegram_bot_user_settings.json"
TRANSCRIPTION_LOG_FILE = Path(__file__).parent / "telegram_bot_transcription_log.json"

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"File di configurazione non trovato: {CONFIG_FILE}")
        print("Crea il file telegram_bot_config.json con il token del bot.")
        sys.exit(1)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_user_settings() -> dict:
    if USER_SETTINGS_FILE.exists():
        with open(USER_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_settings(settings: dict):
    with open(USER_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def load_transcription_log() -> list:
    if TRANSCRIPTION_LOG_FILE.exists():
        with open(TRANSCRIPTION_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_transcription_log(log: list):
    with open(TRANSCRIPTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

CONFIG = load_config()
USER_SETTINGS = load_user_settings()

BOT_TOKEN = CONFIG["telegram_bot_token"]
WHISPER_MODEL = CONFIG.get("whisper_model", "base")
LANGUAGE = CONFIG.get("language", "it")
TASK = CONFIG.get("task", "transcribe")
FP16 = CONFIG.get("fp16", False)
ALLOWED_USERS = CONFIG.get("allowed_user_ids", [])
MAX_FILE_SIZE_MB = CONFIG.get("max_file_size_mb", 2000)
DEFAULT_OUTPUT_FORMAT = CONFIG.get("default_output_format", "message")
HF_TOKEN = CONFIG.get("huggingface_token", "")
DIARIZATION_MODEL = CONFIG.get("diarization_model", "pyannote/speaker-diarization-3.1")

# Output format options: "message" (chat text, live updated), "files", "both"
OUTPUT_FORMATS = {
    "message": "💬 Messaggio in chat (aggiornato in tempo reale)",
    "files": "📄 File .txt (testo puro + con timestamp)",
    "both": "📄+💬 Entrambi",
}

# Available Whisper models (name → description)
WHISPER_MODELS = {
    "tiny": "Tiny (39M) - velocissimo, bassa qualità",
    "base": "Base (74M) - veloce, qualità discreta",
    "small": "Small (244M) - bilanciato",
    "medium": "Medium (769M) - buona qualità",
    "large": "Large (1550M) - qualità massima, lento",
    "large-v2": "Large-v2 - qualità massima v2",
    "large-v3": "Large-v3 - qualità massima v3",
}

# Local API server config
LOCAL_API_CFG = CONFIG.get("local_api_server", {})
USE_LOCAL_API = LOCAL_API_CFG.get("enabled", False)
LOCAL_API_URL = LOCAL_API_CFG.get("api_url", "http://localhost:8081/bot{token}/{method}")
LOCAL_FILE_URL = LOCAL_API_CFG.get("file_url", "http://localhost:8081/file/bot{token}/{path}")
LOCAL_API_DATA_DIR = Path(__file__).parent / "telegram-bot-api-data"

if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    print("Errore: inserisci il token del bot in telegram_bot_config.json")
    sys.exit(1)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Supported formats ---
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".webm", ".ogg", ".wav", ".m4a", ".flac", ".oga", ".mkv", ".avi", ".mov"}


# --- Progress GUI ---
import psutil

# Global job queue for tracking pending transcriptions
transcription_queue: list[dict] = []
transcription_queue_lock = threading.Lock()
transcription_lock: asyncio.Lock | None = None  # initialized at runtime

# GPU monitoring via nvidia-smi
def get_gpu_stats() -> dict | None:
    """Get GPU utilization, memory, temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                return {
                    "gpu_pct": int(parts[0]),
                    "vram_used_mb": int(parts[1]),
                    "vram_total_mb": int(parts[2]),
                    "temp_c": int(parts[3]),
                }
    except Exception:
        pass
    return None


class ProgressGUI:
    """Single persistent tkinter window that reuses itself for each transcription.
    Runs in a dedicated thread. All communication is via thread-safe methods."""

    _instance = None

    @classmethod
    def get(cls) -> "ProgressGUI":
        if cls._instance is None:
            cls._instance = ProgressGUI()
        return cls._instance

    def __init__(self):
        self._thread = None
        self._root = None
        self._ready = threading.Event()
        self._cancel_event: threading.Event | None = None
        self._cancel_source: str | None = None  # 'user', 'server', 'gui'
        self._notify_sound = True
        # Shared state (written from asyncio thread, read from tk thread)
        self._state_lock = threading.Lock()
        self._state = {
            "file_name": "",
            "elapsed": 0,
            "start_time": None,
            "estimate": "",
            "duration": None,
            "text": "In attesa...",
            "status": "idle",  # idle, working, finished, cancelled
            "queue_count": 0,
        }

    def start_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_gui, daemon=True)
            self._thread.start()
            self._ready.wait(timeout=5)

    def begin_transcription(self, file_name: str, time_estimate: str,
                            duration: float | None, cancel_event: threading.Event):
        self._cancel_event = cancel_event
        self._cancel_source = None
        with self._state_lock:
            self._state["file_name"] = file_name
            self._state["estimate"] = time_estimate
            self._state["duration"] = duration
            self._state["start_time"] = time.time()
            self._state["elapsed"] = 0
            self._state["text"] = "In attesa dei primi segmenti..."
            self._state["status"] = "working"
        # Play notification sound
        if self._notify_sound:
            try:
                import winsound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            except Exception:
                pass
        # Restore window if minimized
        if self._root:
            self._root.after(0, self._restore_window)

    def update(self, text: str, elapsed: int):
        with self._state_lock:
            self._state["text"] = text
            self._state["elapsed"] = elapsed

    def mark_finished(self, elapsed: int):
        with self._state_lock:
            self._state["elapsed"] = elapsed
            self._state["status"] = "finished"

    def set_queue_count(self, count: int):
        with self._state_lock:
            self._state["queue_count"] = count

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set() if self._cancel_event else False

    def get_cancel_event(self) -> threading.Event:
        if self._cancel_event is None:
            self._cancel_event = threading.Event()
        return self._cancel_event

    def _restore_window(self):
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.after(100, lambda: self._root.attributes("-topmost", False))

    def _run_gui(self):
        self._root = tk.Tk()
        self._root.title("🎙 Octy Transcribe")
        self._root.geometry("650x550")
        self._root.configure(bg="#1e1e1e")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Header
        header = tk.Frame(self._root, bg="#1e1e1e")
        header.pack(fill=tk.X, padx=15, pady=(15, 5))

        self._title_label = tk.Label(header, text="🎙 In attesa di file...",
                                     font=("Segoe UI", 14, "bold"), fg="#ffffff", bg="#1e1e1e")
        self._title_label.pack(anchor=tk.W)

        self._file_label = tk.Label(header, text="", font=("Segoe UI", 10),
                                    fg="#cccccc", bg="#1e1e1e")
        self._file_label.pack(anchor=tk.W)

        self._queue_label = tk.Label(header, text="", font=("Segoe UI", 9),
                                     fg="#dcdcaa", bg="#1e1e1e")
        self._queue_label.pack(anchor=tk.W)

        # Time
        time_frame = tk.Frame(self._root, bg="#1e1e1e")
        time_frame.pack(fill=tk.X, padx=15, pady=5)

        self._elapsed_label = tk.Label(time_frame, text="",
                                       font=("Segoe UI", 10), fg="#ffffff", bg="#1e1e1e")
        self._elapsed_label.pack(side=tk.LEFT)

        self._estimate_label = tk.Label(time_frame, text="",
                                        font=("Segoe UI", 10), fg="#4ec9b0", bg="#1e1e1e")
        self._estimate_label.pack(side=tk.RIGHT)

        # Progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor="#333333", background="#4ec9b0", thickness=10)
        self._progress = ttk.Progressbar(self._root, mode="indeterminate",
                                         style="Green.Horizontal.TProgressbar")
        self._progress.pack(fill=tk.X, padx=15, pady=5)

        # Resources
        res_frame = tk.Frame(self._root, bg="#252526", relief=tk.FLAT)
        res_frame.pack(fill=tk.X, padx=15, pady=5)

        self._cpu_label = tk.Label(res_frame, text="CPU: ---%", font=("Consolas", 10),
                                   fg="#569cd6", bg="#252526", padx=10, pady=5)
        self._cpu_label.pack(side=tk.LEFT)

        self._ram_label = tk.Label(res_frame, text="RAM: --- GB", font=("Consolas", 10),
                                   fg="#ce9178", bg="#252526", padx=10, pady=5)
        self._ram_label.pack(side=tk.LEFT)

        self._gpu_label = tk.Label(res_frame, text="GPU: ---%", font=("Consolas", 10),
                                   fg="#b5cea8", bg="#252526", padx=10, pady=5)
        self._gpu_label.pack(side=tk.LEFT)

        self._vram_label = tk.Label(res_frame, text="VRAM: --- GB", font=("Consolas", 10),
                                    fg="#dcdcaa", bg="#252526", padx=10, pady=5)
        self._vram_label.pack(side=tk.LEFT)

        # Buttons (must be packed BEFORE the expanding text widget)
        btn_frame = tk.Frame(self._root, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15), side=tk.BOTTOM)

        self._cancel_btn = tk.Button(btn_frame, text="❌ Annulla", font=("Segoe UI", 10),
                                     bg="#c53534", fg="#ffffff", activebackground="#e04040",
                                     activeforeground="#ffffff", relief=tk.FLAT, padx=15, pady=5,
                                     cursor="hand2", command=self._on_cancel, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.RIGHT)

        self._sound_btn = tk.Button(btn_frame, text="🔔 Suono: On", font=("Segoe UI", 10),
                                    bg="#4ec9b0", fg="#1e1e1e", activebackground="#3da890",
                                    activeforeground="#1e1e1e", relief=tk.FLAT, padx=15, pady=5,
                                    cursor="hand2", command=self._toggle_sound)
        self._sound_btn.pack(side=tk.LEFT)

        # Transcript
        tk.Label(self._root, text="Trascrizione:", font=("Segoe UI", 9),
                 fg="#888888", bg="#1e1e1e", anchor=tk.W).pack(fill=tk.X, padx=15, pady=(5, 0))

        text_frame = tk.Frame(self._root, bg="#1e1e1e")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(2, 10))

        self._text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 9),
                                    bg="#252526", fg="#d4d4d4", relief=tk.FLAT, padx=8, pady=8)
        scrollbar = tk.Scrollbar(text_frame, command=self._text_widget.yview)
        self._text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._text_widget.insert("1.0", "Invia un file audio/video al bot per iniziare.")
        self._text_widget.config(state=tk.DISABLED)

        self._ready.set()
        self._tick()
        self._root.mainloop()

    def _on_close(self):
        # Terminate the entire process
        self._root.destroy()
        os._exit(0)

    def _on_cancel(self):
        if self._cancel_event:
            self._cancel_source = 'gui'
            self._cancel_event.set()
        self._cancel_btn.config(text="Annullamento...", state=tk.DISABLED)

    def _toggle_sound(self):
        self._notify_sound = not self._notify_sound
        if self._notify_sound:
            self._sound_btn.config(text="🔔 Suono: On", bg="#4ec9b0", fg="#1e1e1e")
        else:
            self._sound_btn.config(text="🔕 Suono: Off", bg="#555555", fg="#cccccc")

    def _tick(self):
        with self._state_lock:
            s = dict(self._state)

        status = s["status"]

        # Compute elapsed dynamically from start_time
        if status == "working" and s["start_time"]:
            elapsed = int(time.time() - s["start_time"])
        else:
            elapsed = s["elapsed"]

        # Always update resources regardless of status
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            self._cpu_label.config(text=f"CPU: {cpu_pct:.0f}%")
            self._ram_label.config(text=f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB")
        except Exception:
            pass
        try:
            gpu = get_gpu_stats()
            if gpu:
                self._gpu_label.config(text=f"GPU: {gpu['gpu_pct']}% ({gpu['temp_c']}°C)")
                self._vram_label.config(text=f"VRAM: {gpu['vram_used_mb']/1024:.1f}/{gpu['vram_total_mb']/1024:.1f} GB")
        except Exception:
            pass

        if status == "idle":
            self._title_label.config(text="⏸ In attesa di file...")
            self._file_label.config(text="Bot attivo e in ascolto")
            self._elapsed_label.config(text="")
            self._estimate_label.config(text="")
            self._cancel_btn.config(state=tk.DISABLED, text="❌ Annulla", bg="#555555")
            try:
                self._progress.stop()
            except Exception:
                pass
            self._progress.config(mode="determinate", maximum=100)
            self._progress["value"] = 0

        elif status == "working":
            self._title_label.config(text="🎙 Trascrizione in corso")
            self._file_label.config(text=f"File: {s['file_name']}")
            mins = elapsed // 60
            secs = elapsed % 60
            self._elapsed_label.config(text=f"⏱ Trascorso: {mins}:{secs:02d}")
            self._estimate_label.config(text=f"Stima: {s['estimate']}")
            self._cancel_btn.config(state=tk.NORMAL, text="❌ Annulla", bg="#c53534")

            # Progress bar
            if s["duration"] and s["duration"] > 0:
                cur_mode = str(self._progress.cget("mode"))
                if cur_mode == "indeterminate":
                    self._progress.stop()
                    self._progress.config(mode="determinate", maximum=100)
                self._progress["value"] = min(100, (elapsed / s["duration"]) * 100)
            else:
                if str(self._progress.cget("mode")) != "indeterminate":
                    self._progress.config(mode="indeterminate")
                    self._progress.start(30)

        elif status == "finished":
            mins = s["elapsed"] // 60
            secs = s["elapsed"] % 60
            self._title_label.config(text="✅ Trascrizione completata")
            self._file_label.config(text=f"File: {s['file_name']}")
            self._elapsed_label.config(text=f"⏱ Completata in: {mins}:{secs:02d}")
            self._estimate_label.config(text="")
            self._cancel_btn.config(state=tk.DISABLED, text="✅ Fatto", bg="#333333")
            try:
                self._progress.stop()
            except Exception:
                pass
            self._progress.config(mode="determinate", maximum=100)
            self._progress["value"] = 100

        # Queue
        q = s["queue_count"]
        self._queue_label.config(text=f"📋 Coda: {q} file in attesa" if q > 1 else "")

        # Transcript text
        self._text_widget.config(state=tk.NORMAL)
        self._text_widget.delete("1.0", tk.END)
        self._text_widget.insert("1.0", s["text"])
        self._text_widget.see(tk.END)
        self._text_widget.config(state=tk.DISABLED)

        self._root.after(800, self._tick)


def get_user_format(user_id: int) -> str:
    return USER_SETTINGS.get(str(user_id), {}).get("output_format", DEFAULT_OUTPUT_FORMAT)


def set_user_format(user_id: int, fmt: str):
    uid = str(user_id)
    if uid not in USER_SETTINGS:
        USER_SETTINGS[uid] = {}
    USER_SETTINGS[uid]["output_format"] = fmt
    save_user_settings(USER_SETTINGS)


def get_user_model(user_id: int) -> str:
    return USER_SETTINGS.get(str(user_id), {}).get("model", WHISPER_MODEL)


def set_user_model(user_id: int, model: str):
    uid = str(user_id)
    if uid not in USER_SETTINGS:
        USER_SETTINGS[uid] = {}
    USER_SETTINGS[uid]["model"] = model
    save_user_settings(USER_SETTINGS)


def get_user_language(user_id: int) -> str:
    return USER_SETTINGS.get(str(user_id), {}).get("language", LANGUAGE)


def set_user_language(user_id: int, lang: str):
    uid = str(user_id)
    if uid not in USER_SETTINGS:
        USER_SETTINGS[uid] = {}
    USER_SETTINGS[uid]["language"] = lang
    save_user_settings(USER_SETTINGS)


def get_user_diarization(user_id: int) -> bool:
    return USER_SETTINGS.get(str(user_id), {}).get("diarization", False)


def set_user_diarization(user_id: int, enabled: bool):
    uid = str(user_id)
    if uid not in USER_SETTINGS:
        USER_SETTINGS[uid] = {}
    USER_SETTINGS[uid]["diarization"] = enabled
    save_user_settings(USER_SETTINGS)


def estimate_transcription_time(file_size_bytes: int, duration_seconds: float | None, model: str = "base") -> str:
    """Estimate transcription time based on past logs + model loading overhead.
    
    Model: processing_time = load_overhead + duration * speed_ratio
    """
    log = load_transcription_log()

    # Known approximate model speed multipliers (processing_time / audio_duration)
    # and loading overhead in seconds (for RTX 2060 / similar GPU)
    MODEL_DEFAULTS = {
        "tiny":     {"overhead": 5,  "ratio": 0.08},
        "base":     {"overhead": 7,  "ratio": 0.27},
        "small":    {"overhead": 12, "ratio": 0.55},
        "medium":   {"overhead": 18, "ratio": 1.2},
        "large":    {"overhead": 25, "ratio": 2.0},
        "large-v2": {"overhead": 25, "ratio": 2.0},
        "large-v3": {"overhead": 25, "ratio": 2.0},
    }

    # Try to estimate from log entries for this specific model
    model_entries = [e for e in log if e.get("model") == model
                     and e.get("duration_seconds") and e.get("processing_seconds")]

    if len(model_entries) >= 2:
        # Separate short (<30s) and long (>60s) entries to estimate overhead vs speed
        long_entries = [e for e in model_entries if e["duration_seconds"] > 60]
        short_entries = [e for e in model_entries if e["duration_seconds"] <= 30]

        # Estimate overhead from short files (processing time is mostly loading)
        if short_entries:
            overhead = sum(e["processing_seconds"] for e in short_entries) / len(short_entries)
        else:
            overhead = MODEL_DEFAULTS.get(model, MODEL_DEFAULTS["base"])["overhead"]

        # Estimate speed ratio from long files (overhead is negligible fraction)
        if long_entries:
            ratios = [(e["processing_seconds"] - overhead) / e["duration_seconds"]
                      for e in long_entries]
            ratio = max(0.01, sum(ratios) / len(ratios))
        elif len(model_entries) >= 3:
            # Use all entries with simple linear estimation
            ratios = [e["processing_seconds"] / e["duration_seconds"] for e in model_entries]
            ratio = sum(ratios) / len(ratios)
            # Adjust: subtract estimated overhead contribution
            avg_dur = sum(e["duration_seconds"] for e in model_entries) / len(model_entries)
            if avg_dur > 0:
                ratio = max(0.01, ratio - overhead / avg_dur)
        else:
            ratio = MODEL_DEFAULTS.get(model, MODEL_DEFAULTS["base"])["ratio"]

        # Calculate estimate
        if duration_seconds:
            est = overhead + duration_seconds * ratio
        else:
            est_duration = file_size_bytes / 16000
            est = overhead + est_duration * ratio
    else:
        # Fallback to model defaults
        defaults = MODEL_DEFAULTS.get(model, MODEL_DEFAULTS["base"])
        if duration_seconds:
            est = defaults["overhead"] + duration_seconds * defaults["ratio"]
        else:
            est_duration = file_size_bytes / 16000
            est = defaults["overhead"] + est_duration * defaults["ratio"]

    mins = int(est) // 60
    secs = int(est) % 60
    if mins > 0:
        return f"~{mins}m {secs:02d}s"
    return f"~{secs}s"


def estimate_transcription_seconds(file_size_bytes: int, duration_seconds: float | None, model: str = "base") -> float:
    """Return raw estimated seconds (numeric) for queue calculations."""
    text = estimate_transcription_time(file_size_bytes, duration_seconds, model)
    # Parse back from formatted string
    import re as _re
    m = _re.match(r"~(?:(\d+)m\s*)?(\d+)s", text)
    if m:
        mins = int(m.group(1) or 0)
        secs = int(m.group(2))
        return mins * 60 + secs
    return 30  # fallback


def get_audio_duration(file_path: Path) -> float | None:
    """Get audio/video duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def is_user_allowed(user_id: int) -> bool:
    """Se la lista è vuota, tutti sono ammessi."""
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


def parse_srt_to_timed_text(srt_path: Path) -> str:
    """Converte un file .srt in testo con timestamp per riga."""
    if not srt_path.exists():
        return ""
    content = srt_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip sequence number
        if re.match(r"^\d+$", line):
            i += 1
            if i >= len(lines):
                break
            # Timestamp line
            timestamp_line = lines[i].strip()
            match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
            if match:
                start_time = match.group(1).replace(",", ".")
                end_time = match.group(2).replace(",", ".")
                i += 1
                # Collect text lines until empty line
                text_parts = []
                while i < len(lines) and lines[i].strip():
                    text_parts.append(lines[i].strip())
                    i += 1
                text = " ".join(text_parts)
                result.append(f"[{start_time} -> {end_time}] {text}")
            else:
                i += 1
        else:
            i += 1
    return "\n".join(result)


async def run_whisper(media_path: Path, output_dir: Path, segment_callback=None, cancel_event: threading.Event | None = None, model: str | None = None, language: str | None = None) -> tuple[Path | None, Path | None]:
    """Esegue whisper e restituisce (txt_path, srt_path). Chiama segment_callback con ogni segmento trascritto."""
    use_model = model or WHISPER_MODEL
    use_language = language or LANGUAGE
    cmd = [
        sys.executable,
        "-m",
        "whisper",
        str(media_path),
        "--model", use_model,
        "--task", TASK,
        "--output_dir", str(output_dir),
        "--output_format", "all",
        "--verbose", "True",
        "--fp16", str(FP16),
    ]
    if use_language and use_language.lower() != "auto":
        cmd.extend(["--language", use_language])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logger.info(f"Running whisper: {' '.join(cmd)}")

    loop = asyncio.get_event_loop()
    segments_queue: asyncio.Queue = asyncio.Queue()
    whisper_proc = None

    def _run_process():
        nonlocal whisper_proc
        whisper_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        for line in whisper_proc.stdout:
            if cancel_event and cancel_event.is_set():
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(whisper_proc.pid), "/F", "/T"],
                        capture_output=True
                    )
                except Exception:
                    whisper_proc.kill()
                loop.call_soon_threadsafe(segments_queue.put_nowait, None)
                return -1
            line = line.rstrip("\r\n")
            if line and re.match(r"^\[\d{2}:\d{2}", line):
                loop.call_soon_threadsafe(segments_queue.put_nowait, line)
        whisper_proc.wait()
        loop.call_soon_threadsafe(segments_queue.put_nowait, None)  # sentinel
        return whisper_proc.returncode

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_process)

    # Process segments as they come in
    all_segments = []
    last_update_time = 0
    while True:
        # Check cancellation
        if cancel_event and cancel_event.is_set():
            # Wait briefly for process to be assigned by thread
            for _ in range(20):
                if whisper_proc is not None:
                    break
                await asyncio.sleep(0.05)
            if whisper_proc and whisper_proc.poll() is None:
                # Kill entire process tree on Windows
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(whisper_proc.pid), "/F", "/T"],
                        capture_output=True
                    )
                except Exception:
                    whisper_proc.kill()
                # Close stdout to unblock the reader thread
                try:
                    whisper_proc.stdout.close()
                except Exception:
                    pass
            break

        try:
            segment = await asyncio.wait_for(segments_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if future.done():
                break
            continue

        if segment is None:
            break

        all_segments.append(segment)
        now = time.time()
        # Update at most every 3 seconds to respect Telegram rate limits
        if segment_callback and (now - last_update_time >= 3.0):
            last_update_time = now
            await segment_callback(all_segments)

    # Final update with all segments
    if segment_callback and all_segments:
        await segment_callback(all_segments)

    if cancel_event and cancel_event.is_set():
        executor.shutdown(wait=False)
        return None, None

    return_code = future.result()
    executor.shutdown(wait=False)

    if return_code != 0:
        logger.error(f"Whisper failed with return code {return_code}")
        return None, None

    stem = media_path.stem
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"

    if not txt_path.exists():
        txt_path = None
    if not srt_path.exists():
        srt_path = None

    return txt_path, srt_path


async def run_diarization(media_path: Path, output_path: Path, cancel_event: threading.Event | None = None) -> Path | None:
    """Run speaker diarization using the diarization.py script."""
    diarization_script = Path(__file__).parent / "diarization.py"
    if not diarization_script.exists():
        logger.error(f"Diarization script not found: {diarization_script}")
        return None

    if not HF_TOKEN:
        logger.error("HuggingFace token not configured for diarization")
        return None

    cmd = [
        sys.executable,
        str(diarization_script),
        "--token", HF_TOKEN,
        "--model", DIARIZATION_MODEL,
        "--audio_file", str(media_path),
        "--diarization_output", str(output_path),
        "--use_gpu",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logger.info(f"Running diarization: {' '.join(cmd[:6])}...")

    import concurrent.futures
    loop = asyncio.get_event_loop()

    def _run():
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        for line in proc.stdout:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                return -1
            logger.debug(f"[diarization] {line.rstrip()}")
        proc.wait()
        return proc.returncode

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run)

    while not future.done():
        if cancel_event and cancel_event.is_set():
            break
        await asyncio.sleep(1)

    return_code = future.result()
    executor.shutdown(wait=False)

    if return_code != 0 or not output_path.exists():
        logger.error(f"Diarization failed (code={return_code})")
        return None

    return output_path


async def run_merge_diarization(diarization_path: Path, srt_path: Path, output_path: Path) -> Path | None:
    """Merge diarization results with SRT transcript."""
    merge_script = Path(__file__).parent / "merge-transcript-diarization.py"
    if not merge_script.exists():
        logger.error(f"Merge script not found: {merge_script}")
        return None

    cmd = [
        sys.executable,
        str(merge_script),
        "--diarization_file", str(diarization_path),
        "--srt_file", str(srt_path),
        "--output_file", str(output_path),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(cmd, capture_output=True, text=True, env=env)
    )

    if result.returncode != 0:
        logger.error(f"Merge failed: {result.stderr}")
        return None

    if output_path.exists():
        return output_path
    return None


# --- Bot handlers ---

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_fmt = get_user_format(uid)
    user_model = get_user_model(uid)
    user_lang = get_user_language(uid)
    mode = "Local API (no limiti)" if USE_LOCAL_API else "Cloud API (max ~20 MB)"
    await update.message.reply_text(
        "🎙 Bot Trascrizione Audio/Video\n\n"
        "Inviami un file audio o video e ti restituirò la trascrizione.\n\n"
        f"Output: {OUTPUT_FORMATS[user_fmt]}\n"
        f"Modello: {WHISPER_MODELS.get(user_model, user_model)}\n"
        f"Lingua: {user_lang}\n"
        f"Modalità: {mode}\n\n"
        "Comandi:\n"
        "/start - Mostra questo messaggio\n"
        "/settings - Impostazioni personali\n"
        "/status - Stato del bot"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_fmt = get_user_format(uid)
    user_model = get_user_model(uid)
    user_lang = get_user_language(uid)
    gpu = get_gpu_stats()
    gpu_text = f"GPU: {gpu['gpu_pct']}% | VRAM: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB | {gpu['temp_c']}°C" if gpu else "GPU: N/A"
    await update.message.reply_text(
        f"✅ Bot online\n\n"
        f"🧠 Modello: {WHISPER_MODELS.get(user_model, user_model)}\n"
        f"🌍 Lingua: {user_lang}\n"
        f"📤 Output: {OUTPUT_FORMATS[user_fmt]}\n"
        f"📦 Max file: {MAX_FILE_SIZE_MB} MB\n\n"
        f"💻 {gpu_text}\n"
        f"🌐 Dashboard: http://localhost:3001"
    )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_fmt = get_user_format(uid)
    user_model = get_user_model(uid)
    user_lang = get_user_language(uid)
    user_diar = get_user_diarization(uid)

    keyboard = []
    # Output format section
    keyboard.append([InlineKeyboardButton("── 📤 Formato Output ──", callback_data="noop")])
    for key, label in OUTPUT_FORMATS.items():
        check = " ✓" if key == user_fmt else ""
        keyboard.append([InlineKeyboardButton(f"{label}{check}", callback_data=f"fmt:{key}")])

    # Model section
    keyboard.append([InlineKeyboardButton("── 🧠 Modello Whisper ──", callback_data="noop")])
    for key, label in WHISPER_MODELS.items():
        check = " ✓" if key == user_model else ""
        keyboard.append([InlineKeyboardButton(f"{label}{check}", callback_data=f"mdl:{key}")])

    # Language section
    keyboard.append([InlineKeyboardButton("── 🌍 Lingua ──", callback_data="noop")])
    LANGUAGES = {"it": "🇮🇹 Italiano", "en": "🇬🇧 English", "auto": "🔄 Auto-detect"}
    for key, label in LANGUAGES.items():
        check = " ✓" if key == user_lang else ""
        keyboard.append([InlineKeyboardButton(f"{label}{check}", callback_data=f"lng:{key}")])

    # Diarization section
    keyboard.append([InlineKeyboardButton("── 🗣️ Diarizzazione ──", callback_data="noop")])
    diar_status = "✅ Attiva" if user_diar else "❌ Disattiva"
    toggle_label = "🗣️ Disattiva diarizzazione" if user_diar else "🗣️ Attiva diarizzazione"
    keyboard.append([InlineKeyboardButton(f"{toggle_label}", callback_data="diar:toggle")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "⚙️ **Impostazioni personali**\n\n"
        f"📤 Formato: {OUTPUT_FORMATS[user_fmt]}\n"
        f"🧠 Modello: {WHISPER_MODELS.get(user_model, user_model)}\n"
        f"🌍 Lingua: {user_lang}\n"
        f"🗣️ Diarizzazione: {diar_status}\n\n"
        "Seleziona un'opzione per modificare:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "noop":
        return

    if data.startswith("fmt:"):
        fmt = data[4:]
        if fmt in OUTPUT_FORMATS:
            set_user_format(query.from_user.id, fmt)
            await query.edit_message_text(
                f"✅ Formato output aggiornato: {OUTPUT_FORMATS[fmt]}\n\n"
                "La prossima trascrizione userà questo formato."
            )

    elif data.startswith("mdl:"):
        model = data[4:]
        if model in WHISPER_MODELS:
            set_user_model(query.from_user.id, model)
            await query.edit_message_text(
                f"✅ Modello aggiornato: {WHISPER_MODELS[model]}\n\n"
                "La prossima trascrizione userà questo modello.\n"
                "⚠️ Modelli più grandi richiedono più VRAM e tempo."
            )

    elif data.startswith("lng:"):
        lang = data[4:]
        set_user_language(query.from_user.id, lang)
        await query.edit_message_text(
            f"✅ Lingua aggiornata: {lang}\n\n"
            "La prossima trascrizione userà questa lingua."
        )

    elif data.startswith("diar:"):
        current = get_user_diarization(query.from_user.id)
        new_val = not current
        set_user_diarization(query.from_user.id, new_val)
        status = "attivata ✅" if new_val else "disattivata ❌"
        note = ""
        if new_val and not HF_TOKEN:
            note = "\n\n⚠️ Token HuggingFace non configurato nel server. La diarizzazione potrebbe non funzionare."
        await query.edit_message_text(
            f"✅ Diarizzazione {status}\n\n"
            "Dopo la trascrizione verrà identificato chi parla "
            f"e il risultato sarà un file SRT con i nomi degli speaker.{note}"
        )


async def cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline cancel button pressed by user during transcription."""
    query = update.callback_query
    gui = ProgressGUI.get()
    if gui._cancel_event and not gui._cancel_event.is_set():
        gui._cancel_source = 'user'
        gui._cancel_event.set()
        await query.answer("⏹ Annullamento in corso...")
    else:
        await query.answer("Nessuna trascrizione in corso.", show_alert=True)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce file audio, video, voice e video_note."""
    user = update.effective_user
    if not is_user_allowed(user.id):
        await update.message.reply_text("⛔ Non sei autorizzato ad usare questo bot.")
        return

    # Determine file object
    message = update.message
    file_obj = None
    file_name = None

    if message.document:
        file_obj = message.document
        file_name = message.document.file_name or "document"
    elif message.audio:
        file_obj = message.audio
        file_name = message.audio.file_name or "audio.mp3"
    elif message.video:
        file_obj = message.video
        file_name = message.video.file_name or "video.mp4"
    elif message.voice:
        file_obj = message.voice
        file_name = "voice.ogg"
    elif message.video_note:
        file_obj = message.video_note
        file_name = "video_note.mp4"

    if not file_obj:
        await message.reply_text("❌ Formato non riconosciuto. Invia un file audio o video.")
        return

    # Check extension
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        await message.reply_text(
            f"❌ Formato '{ext}' non supportato.\n"
            f"Formati accettati: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    # Check size
    file_size_bytes = file_obj.file_size or 0
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        await message.reply_text(
            f"❌ File troppo grande ({file_size_mb:.1f} MB).\n"
            f"Massimo consentito: {MAX_FILE_SIZE_MB} MB"
        )
        return

    user_fmt = get_user_format(user.id)
    user_model = get_user_model(user.id)
    user_lang = get_user_language(user.id)
    user_diar = get_user_diarization(user.id)
    queue_entry = {"file_name": file_name, "user_name": user.first_name or "User", "queued_at": time.time(), "estimated_seconds": 0}
    with transcription_queue_lock:
        transcription_queue.append(queue_entry)

    diar_label = "🗣️ Diarizzazione: Sì" if user_diar else "🗣️ Diarizzazione: No"

    # Estimate time for queue display
    est_seconds = estimate_transcription_seconds(file_size_bytes, None, model=user_model)
    queue_entry["estimated_seconds"] = est_seconds

    # If another transcription is running, notify with queue info
    if transcription_lock and transcription_lock.locked():
        with transcription_queue_lock:
            queue_count = len(transcription_queue)
            total_est = sum(q.get("estimated_seconds", 0) for q in transcription_queue[:-1])
        if total_est >= 60:
            wait_text = f"~{int(total_est)//60}m {int(total_est)%60:02d}s"
        else:
            wait_text = f"~{int(total_est)}s"
        await message.reply_text(
            f"⏳ In coda: posizione {queue_count} | Attesa stimata: {wait_text}"
        )

    # Download and process
    status_msg = await message.reply_text(
        f"📥 Scaricamento file ({file_size_mb:.1f} MB)...\n"
        f"Modello: {user_model} | Lingua: {user_lang}\n"
        f"{diar_label}"
    )

    await transcription_lock.acquire()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Sanitize filename
            safe_name = re.sub(r'[^\w\-.]', '_', file_name)
            download_path = tmp_path / safe_name

            try:
                tg_file = await file_obj.get_file(read_timeout=600, write_timeout=600, connect_timeout=30)

                if USE_LOCAL_API and tg_file.file_path:
                    raw_path = tg_file.file_path
                    if "/var/lib/telegram-bot-api/" in raw_path:
                        container_path = "/var/lib/telegram-bot-api/" + raw_path.split("/var/lib/telegram-bot-api/", 1)[1]
                    else:
                        await tg_file.download_to_drive(str(download_path), read_timeout=600, write_timeout=600, connect_timeout=30)
                        container_path = None

                    if container_path:
                        logger.info(f"Copying from container: {container_path}")
                        docker_cmd = "docker"
                        for docker_path in [
                            r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
                            "docker",
                        ]:
                            if Path(docker_path).exists() or docker_path == "docker":
                                docker_cmd = docker_path
                                break
                        proc = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: subprocess.run(
                                [docker_cmd, "cp",
                                 f"recordings-telegram-bot-api-1:{container_path}",
                                 str(download_path)],
                                capture_output=True, text=True
                            )
                        )
                        if proc.returncode != 0:
                            raise Exception(f"docker cp failed: {proc.stderr}")
                else:
                    await tg_file.download_to_drive(str(download_path), read_timeout=600, write_timeout=600, connect_timeout=30)
            except Exception as dl_err:
                if "file is too big" in str(dl_err).lower():
                    await status_msg.edit_text(
                        "❌ File troppo grande per il download.\n\n"
                        "Telegram limita il download via bot a ~20 MB (API cloud).\n"
                        "Soluzioni:\n"
                        "• Comprimi il file prima di inviarlo\n"
                        "• Invia un file più corto\n"
                        "• Converti in formato audio (mp3/ogg) che pesa meno"
                    )
                else:
                    await status_msg.edit_text(f"❌ Errore download: {dl_err}")
                with transcription_queue_lock:
                    if queue_entry in transcription_queue:
                        transcription_queue.remove(queue_entry)
                return

            # Get audio duration for time estimate
            duration = get_audio_duration(download_path)
            time_est = estimate_transcription_time(file_size_bytes, duration, model=user_model)

            steps = [
                "📥 Download",
                "⏳ Caricamento modello",
                "🎙️ Trascrizione",
            ]
            if user_diar:
                steps.append("🗣️ Diarizzazione")
            steps_text = " → ".join(steps)

            cancel_kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Annulla trascrizione", callback_data="cancel_transcription")]
            ])

            await status_msg.edit_text(
                f"⏳ Caricamento modello Whisper ({user_model})...\n\n"
                f"📋 Pipeline: {steps_text}\n"
                f"⏱️ Stima: {time_est}\n"
                f"{diar_label}",
                reply_markup=cancel_kb
            )

            # Spawn progress GUI window
            gui = ProgressGUI.get()
            gui.start_thread()
            cancel_event = threading.Event()
            gui.begin_transcription(file_name, time_est, duration, cancel_event)
            with transcription_queue_lock:
                gui.set_queue_count(len(transcription_queue))

            # Message for live transcript (only if message or both mode)
            live_msg = None
            if user_fmt in ("message", "both"):
                live_msg = await message.reply_text("📝 ...")

            start_time_proc = time.time()

            # Segment callback: update message with transcript as it's generated
            last_text_sent = ""
            first_segment_received = False

            async def on_segments(segments: list[str]):
                nonlocal last_text_sent, first_segment_received

                # On first segment, update status to "transcribing"
                if not first_segment_received:
                    first_segment_received = True
                    try:
                        await status_msg.edit_text(
                            f"🎙️ Trascrizione in corso ({user_model})...\n"
                            f"Segmenti: {len(segments)} | "
                            f"Tempo: {int(time.time() - start_time_proc)}s\n"
                            f"{diar_label}",
                            reply_markup=cancel_kb
                        )
                    except Exception:
                        pass

                # Build transcript text from segments (extract just the text part)
                lines = []
                for seg in segments:
                    # seg format: [00:00.000 --> 00:05.000]  Some text here
                    m = re.match(r"^\[\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}\]\s*(.*)", seg)
                    if m:
                        lines.append(m.group(1).strip())
                    else:
                        lines.append(seg)
                text = " ".join(lines)

                # Update GUI window
                elapsed = int(time.time() - start_time_proc)
                gui.update(text, elapsed)

                # Update status message with progress (every callback update)
                try:
                    await status_msg.edit_text(
                        f"🎙️ Trascrizione in corso ({user_model})...\n"
                        f"Segmenti: {len(segments)} | "
                        f"Tempo: {elapsed}s\n"
                        f"{diar_label}",
                        reply_markup=cancel_kb
                    )
                except Exception:
                    pass

                # Update Telegram message
                if live_msg:
                    display = text
                    if len(display) > 4000:
                        display = "..." + display[-4000:]
                    if display != last_text_sent:
                        last_text_sent = display
                        try:
                            await live_msg.edit_text(f"📝 {display}")
                        except Exception:
                            pass  # rate limit or unchanged

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            txt_path, srt_path = await run_whisper(download_path, output_dir,
                                                   segment_callback=on_segments,
                                                   cancel_event=cancel_event,
                                                   model=user_model,
                                                   language=user_lang)

            processing_time = time.time() - start_time_proc

            # Update status after whisper completes
            if not cancel_event.is_set() and txt_path:
                if user_diar:
                    try:
                        await status_msg.edit_text(
                            f"✅ Trascrizione completata ({int(processing_time)}s)\n"
                            f"🗣️ Avvio diarizzazione speaker..."
                        )
                    except Exception:
                        pass
                else:
                    try:
                        await status_msg.edit_text(
                            f"✅ Trascrizione completata ({int(processing_time)}s)\n"
                            f"📤 Invio risultati..."
                        )
                    except Exception:
                        pass

            gui.mark_finished(int(processing_time))

            # Remove from queue
            with transcription_queue_lock:
                if queue_entry in transcription_queue:
                    transcription_queue.remove(queue_entry)

            # Handle cancellation
            if cancel_event.is_set():
                source = gui._cancel_source or 'user'
                if source == 'server' or source == 'gui':
                    cancel_text = "🚫 Trascrizione annullata dal server."
                else:
                    cancel_text = "🚫 Trascrizione annullata dall'utente."
                await status_msg.edit_text(cancel_text)
                if live_msg:
                    try:
                        await live_msg.delete()
                    except Exception:
                        pass
                return

            if txt_path is None:
                await status_msg.edit_text("❌ Errore durante la trascrizione. Controlla i log del server.")
                if live_msg:
                    await live_msg.delete()
                return

            # Log transcription for future estimates
            log = load_transcription_log()
            log.append({
                "file_name": file_name,
                "file_size_bytes": file_size_bytes,
                "duration_seconds": duration,
                "processing_seconds": round(processing_time, 1),
                "model": user_model,
                "timestamp": time.time(),
            })
            # Keep last 100 entries
            if len(log) > 100:
                log = log[-100:]
            save_transcription_log(log)

            # Read final transcript
            plain_text = txt_path.read_text(encoding="utf-8").strip()
            if not plain_text:
                await status_msg.edit_text("⚠️ La trascrizione è vuota. Il file potrebbe non contenere audio parlato.")
                if live_msg:
                    await live_msg.delete()
                return

            timed_text = ""
            if srt_path and srt_path.exists():
                timed_text = parse_srt_to_timed_text(srt_path)

            # Final output based on user format
            if user_fmt in ("message", "both"):
                # Update live message with final complete transcript
                if live_msg:
                    try:
                        if len(plain_text) <= 4000:
                            await live_msg.edit_text(f"📝 {plain_text}")
                        else:
                            # Too long for a single message, split or send as file
                            await live_msg.edit_text(
                                f"📝 Trascrizione troppo lunga per un messaggio ({len(plain_text)} caratteri). "
                                "Invio come file..."
                            )
                            # Send as file anyway
                            base_name = Path(file_name).stem
                            plain_output = tmp_path / f"{base_name}_transcript.txt"
                            plain_output.write_text(plain_text, encoding="utf-8")
                            await message.reply_document(
                                document=open(plain_output, "rb"),
                                filename=plain_output.name,
                                caption="📄 Trascrizione (testo puro)",
                            )
                    except Exception:
                        pass  # message already has this content

            if user_fmt in ("files", "both"):
                base_name = Path(file_name).stem
                plain_output = tmp_path / f"{base_name}_transcript.txt"
                timed_output = tmp_path / f"{base_name}_transcript_timed.txt"
                plain_output.write_text(plain_text, encoding="utf-8")
                timed_output.write_text(timed_text or plain_text, encoding="utf-8")

                await message.reply_document(
                    document=open(plain_output, "rb"),
                    filename=plain_output.name,
                    caption="📄 Trascrizione (testo puro)",
                )
                await message.reply_document(
                    document=open(timed_output, "rb"),
                    filename=timed_output.name,
                    caption="🕐 Trascrizione (con timestamp)",
                )

            # --- Diarization ---
            if user_diar and srt_path and srt_path.exists():
                try:
                    await status_msg.edit_text(
                        f"🗣️ Diarizzazione in corso (identificazione speaker)..."
                    )
                    gui.update("🗣️ Diarizzazione in corso...", int(time.time() - start_time_proc))

                    diar_output = tmp_path / f"{Path(file_name).stem}_diarization.txt"
                    diar_result = await run_diarization(download_path, diar_output, cancel_event)

                    if cancel_event.is_set():
                        return

                    if diar_result:
                        # Merge with SRT
                        speaker_srt = tmp_path / f"{Path(file_name).stem}_speaker.srt"
                        merge_result = await run_merge_diarization(diar_result, srt_path, speaker_srt)

                        if merge_result and merge_result.exists():
                            await message.reply_document(
                                document=open(merge_result, "rb"),
                                filename=speaker_srt.name,
                                caption="🗣️ Trascrizione con speaker (SRT)",
                            )
                            # Also send plain diarization
                            await message.reply_document(
                                document=open(diar_result, "rb"),
                                filename=diar_output.name,
                                caption="🗣️ Segmenti diarizzazione",
                            )
                        else:
                            await message.reply_text("⚠️ Merge diarizzazione fallito, invio solo segmenti.")
                            await message.reply_document(
                                document=open(diar_result, "rb"),
                                filename=diar_output.name,
                                caption="🗣️ Segmenti diarizzazione",
                            )
                    else:
                        await message.reply_text(
                            "⚠️ Diarizzazione fallita. Verifica che il token HuggingFace sia configurato."
                        )
                except Exception as diar_err:
                    logger.exception("Errore diarizzazione")
                    await message.reply_text(f"⚠️ Errore diarizzazione: {diar_err}")

            total_time = time.time() - start_time_proc
            total_mins = int(total_time) // 60
            total_secs = int(total_time) % 60

            try:
                await status_msg.edit_text(
                    f"✅ Trascrizione completata in {total_mins}m {total_secs:02d}s"
                )
            except Exception:
                pass

    except Exception as e:
        logger.exception("Errore durante la trascrizione")
        with transcription_queue_lock:
            if queue_entry in transcription_queue:
                transcription_queue.remove(queue_entry)
        try:
            await status_msg.edit_text(f"❌ Errore: {e}")
        except Exception:
            pass
    finally:
        transcription_lock.release()


async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Inviami un file audio o video per la trascrizione.\n"
        "Digita /start per maggiori info."
    )


# --- HTTP API for React dashboard ---
from aiohttp import web
import aiohttp

WEB_PORT = 3001
_ws_clients: list[web.WebSocketResponse] = []


async def ws_broadcast(data: dict):
    """Broadcast JSON message to all connected WebSocket clients."""
    msg = json.dumps(data)
    for ws in list(_ws_clients):
        try:
            await ws.send_str(msg)
        except Exception:
            _ws_clients.remove(ws)


def _get_status_payload() -> dict:
    """Build current status payload for API/WS."""
    gui = ProgressGUI.get()
    with gui._state_lock:
        s = dict(gui._state)

    # Compute elapsed dynamically
    if s["status"] == "working" and s.get("start_time"):
        elapsed = int(time.time() - s["start_time"])
    else:
        elapsed = s["elapsed"]

    with transcription_queue_lock:
        queue_list = [
            {"file_name": q["file_name"], "user_name": q.get("user_name", "?"),
             "queued_at": q.get("queued_at", 0)}
            for q in transcription_queue
        ]

    gpu = get_gpu_stats()
    cpu_pct = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()

    return {
        "status": s["status"],
        "file_name": s["file_name"],
        "elapsed": elapsed,
        "estimate": s["estimate"],
        "duration": s["duration"],
        "text": s["text"],
        "queue": queue_list,
        "queue_count": len(queue_list),
        "system": {
            "cpu_pct": cpu_pct,
            "ram_used_gb": round(ram.used / (1024**3), 1),
            "ram_total_gb": round(ram.total / (1024**3), 1),
            "gpu": gpu,
        },
        "config": {
            "model": WHISPER_MODEL,
            "language": LANGUAGE,
            "task": TASK,
        },
    }


async def api_status(request):
    return web.json_response(_get_status_payload())


async def api_cancel(request):
    gui = ProgressGUI.get()
    if gui._cancel_event and not gui._cancel_event.is_set():
        gui._cancel_source = 'server'
        gui._cancel_event.set()
        return web.json_response({"ok": True, "message": "Cancellation requested"})
    return web.json_response({"ok": False, "message": "Nothing to cancel"})


async def api_log(request):
    log = load_transcription_log()
    return web.json_response(log[-50:])


async def api_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.append(ws)
    try:
        # Send initial status
        await ws.send_str(json.dumps(_get_status_payload()))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if msg.data == "status":
                    await ws.send_str(json.dumps(_get_status_payload()))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
    return ws


async def start_web_server():
    """Start the aiohttp web server for the React dashboard."""
    # CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def _build_app():
        app_web = web.Application(middlewares=[cors_middleware])
        app_web.router.add_get("/api/status", api_status)
        app_web.router.add_post("/api/cancel", api_cancel)
        app_web.router.add_get("/api/log", api_log)
        app_web.router.add_get("/api/ws", api_ws)

        # Serve React build
        frontend_dist = Path(__file__).parent / "frontend" / "dist"
        if frontend_dist.exists():
            app_web.router.add_static("/assets", frontend_dist / "assets")
            app_web.router.add_get("/", lambda r: web.FileResponse(frontend_dist / "index.html"))
            app_web.router.add_get("/{path:.*}", lambda r: web.FileResponse(frontend_dist / "index.html"))
        return app_web

    async def _try_start():
        app_web = _build_app()
        runner = web.AppRunner(app_web)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", WEB_PORT)
        await site.start()
        return True

    try:
        await _try_start()
    except OSError as e:
        if e.errno in (10048, 98):  # EADDRINUSE on Windows/Linux
            logger.warning(f"Porta {WEB_PORT} in uso, tentativo di liberarla...")
            await _kill_port(WEB_PORT)
            await asyncio.sleep(2)
            try:
                await _try_start()
            except OSError:
                logger.error(f"Impossibile avviare dashboard su porta {WEB_PORT}. Continuazione senza dashboard.")
                return
        else:
            raise
    logger.info(f"Dashboard web avviata su http://localhost:{WEB_PORT}")


async def _kill_port(port: int):
    """Kill any process holding the given port."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True
            )
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    logger.info(f"Killing PID {pid} holding port {port}")
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: subprocess.run(
                            ["taskkill", "/PID", pid, "/F"],
                            capture_output=True
                        )
                    )
    except Exception as e:
        logger.warning(f"Errore tentativo kill porta: {e}")


# Background task to broadcast status via WebSocket every 2s
async def ws_broadcast_loop():
    while True:
        await asyncio.sleep(2)
        if _ws_clients:
            try:
                await ws_broadcast(_get_status_payload())
            except Exception:
                pass


def main():
    global transcription_lock
    logger.info("Avvio bot trascrizione...")
    logger.info(f"Modello: {WHISPER_MODEL}, Lingua: {LANGUAGE}, Task: {TASK}")

    transcription_lock = asyncio.Lock()

    builder = Application.builder().token(BOT_TOKEN).concurrent_updates(True)

    # Increase timeouts for large file downloads (default is 5s read, 5s write)
    builder = builder.read_timeout(600).write_timeout(600).connect_timeout(30)

    if USE_LOCAL_API:
        base_url = "http://localhost:8081/bot"
        base_file_url = "http://localhost:8081/file/bot"
        builder = builder.base_url(base_url).base_file_url(base_file_url)
        logger.info(f"Usando Local Bot API Server: {base_url}")
    else:
        logger.info("Usando Telegram Cloud API (limite download: 20 MB)")

    app = builder.build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("settings", cmd_settings))

    # Settings callback
    app.add_handler(CallbackQueryHandler(settings_callback, pattern="^(fmt:|mdl:|lng:|diar:|noop)"))

    # Cancel callback (user presses inline cancel button)
    app.add_handler(CallbackQueryHandler(cancel_callback, pattern="^cancel_transcription$"))

    # Media handlers
    app.add_handler(MessageHandler(filters.AUDIO, handle_media))
    app.add_handler(MessageHandler(filters.VIDEO, handle_media))
    app.add_handler(MessageHandler(filters.VOICE, handle_media))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_media))
    app.add_handler(MessageHandler(
        filters.Document.ALL & (
            filters.Document.MimeType("audio/mpeg") |
            filters.Document.MimeType("audio/mp4") |
            filters.Document.MimeType("audio/ogg") |
            filters.Document.MimeType("audio/wav") |
            filters.Document.MimeType("audio/x-wav") |
            filters.Document.MimeType("audio/flac") |
            filters.Document.MimeType("audio/x-m4a") |
            filters.Document.MimeType("video/mp4") |
            filters.Document.MimeType("video/webm") |
            filters.Document.MimeType("video/x-matroska") |
            filters.Document.MimeType("video/quicktime") |
            filters.Document.MimeType("video/x-msvideo") |
            filters.Document.FileExtension("mp3") |
            filters.Document.FileExtension("mp4") |
            filters.Document.FileExtension("webm") |
            filters.Document.FileExtension("ogg") |
            filters.Document.FileExtension("wav") |
            filters.Document.FileExtension("m4a") |
            filters.Document.FileExtension("flac") |
            filters.Document.FileExtension("mkv") |
            filters.Document.FileExtension("avi") |
            filters.Document.FileExtension("mov")
        ),
        handle_media,
    ))

    # Fallback
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))

    # Start GUI window immediately
    gui = ProgressGUI.get()
    gui.start_thread()

    # Start web dashboard in post_init
    async def post_init(application):
        await start_web_server()
        asyncio.create_task(ws_broadcast_loop())
        # Set bot menu commands
        from telegram import BotCommand
        await application.bot.set_my_commands([
            BotCommand("start", "Info sul bot"),
            BotCommand("status", "Stato e configurazione"),
            BotCommand("settings", "Impostazioni (modello, lingua, formato, diarizzazione)"),
        ])

    app.post_init = post_init

    # Long polling
    logger.info("Bot avviato in modalità polling. Premi Ctrl+C per fermare.")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
