import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import subprocess
import importlib.util
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

SUPPORTED_EXTENSIONS = {".mp3", ".webm", ".mp4"}
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_LANGUAGE_OPTIONS = [
    "auto",
    "it",
    "en",
    "es",
    "fr",
    "de",
    "pt",
    "nl",
    "ru",
    "ja",
    "zh",
]
APP_DIR = Path(__file__).resolve().parent
APP_STATE_FILE = APP_DIR / ".transcriber_ui_state.json"
SECRETS_FILE = APP_DIR / ".transcriber_secrets.json"


class TranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Media Transcriber")
        self.root.resizable(False, False)

        # Set window icon
        ico_path = Path(__file__).parent / "app.ico"
        if ico_path.exists():
            self.root.iconbitmap(str(ico_path))

        self.running = False
        self.cancel_requested = False
        self.current_process = None
        self.media_files: list[Path] = []
        self.last_log_was_progress = False

        self._build_ui()
        self._load_saved_state()
        self._load_files()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # --- Input folder ---
        frame_dir = ttk.LabelFrame(self.root, text="Input Folder", padding=8)
        frame_dir.pack(fill="x", padx=10, pady=(10, 4))

        self.dir_var = tk.StringVar(value=str(Path(".").resolve()))
        ttk.Entry(frame_dir, textvariable=self.dir_var, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(frame_dir, text="Browse…", command=self._browse_folder).pack(side="left", padx=(6, 0))

        # --- File list ---
        frame_files = ttk.LabelFrame(self.root, text="Files", padding=8)
        frame_files.pack(fill="both", expand=True, padx=10, pady=4)

        list_frame = ttk.Frame(frame_files)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.file_list = tk.Listbox(list_frame, selectmode="extended", height=10, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.file_list.yview)
        self.file_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_frame = ttk.Frame(frame_files)
        btn_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_frame, text="Refresh", command=self._load_files).pack(side="left")
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side="left", padx=(6, 0))
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all).pack(side="left", padx=(6, 0))

        # --- Options ---
        frame_opts = ttk.LabelFrame(self.root, text="Options", padding=8)
        frame_opts.pack(fill="x", padx=10, pady=4)

        self.do_transcription_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_opts, text="Run transcription", variable=self.do_transcription_var).pack(anchor="w")

        self.skip_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_opts, text="Skip existing output files", variable=self.skip_existing_var).pack(anchor="w")

        transcribe_model_row = ttk.Frame(frame_opts)
        transcribe_model_row.pack(anchor="w", pady=(2, 0))
        ttk.Label(transcribe_model_row, text="Whisper model:").pack(side="left", padx=(0, 4))
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(transcribe_model_row, textvariable=self.model_var, state="readonly", width=12,
                                   values=["tiny", "base", "small", "medium", "large"])
        model_combo.pack(side="left")

        language_row = ttk.Frame(frame_opts)
        language_row.pack(anchor="w", pady=(2, 0))
        ttk.Label(language_row, text="Language:").pack(side="left", padx=(0, 4))
        self.language_var = tk.StringVar(value="auto")
        language_combo = ttk.Combobox(
            language_row,
            textvariable=self.language_var,
            state="readonly",
            width=12,
            values=WHISPER_LANGUAGE_OPTIONS,
        )
        language_combo.pack(side="left")

        ttk.Separator(frame_opts, orient="horizontal").pack(fill="x", pady=6)

        self.do_diarization_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_opts, text="Run diarization", variable=self.do_diarization_var).pack(anchor="w")

        diar_model_row = ttk.Frame(frame_opts)
        diar_model_row.pack(anchor="w", pady=(2, 0))
        ttk.Label(diar_model_row, text="Diarization model:").pack(side="left", padx=(0, 4))
        self.diar_model_var = tk.StringVar(value=DEFAULT_DIARIZATION_MODEL)
        ttk.Entry(diar_model_row, textvariable=self.diar_model_var, width=34).pack(side="left")

        token_row = ttk.Frame(frame_opts)
        token_row.pack(anchor="w", pady=(2, 0))
        ttk.Label(token_row, text="HF token:").pack(side="left", padx=(0, 4))
        self.hf_token_var = tk.StringVar(value=os.environ.get("HF_TOKEN", ""))
        ttk.Entry(token_row, textvariable=self.hf_token_var, width=34, show="*").pack(side="left")

        diar_opts_row = ttk.Frame(frame_opts)
        diar_opts_row.pack(anchor="w", pady=(2, 0))
        self.use_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(diar_opts_row, text="Use GPU", variable=self.use_gpu_var).pack(side="left")
        ttk.Label(diar_opts_row, text="Num speakers:").pack(side="left", padx=(10, 4))
        self.num_speakers_var = tk.StringVar(value="")
        ttk.Entry(diar_opts_row, textvariable=self.num_speakers_var, width=6).pack(side="left")

        # --- Progress ---
        frame_prog = ttk.LabelFrame(self.root, text="Progress", padding=8)
        frame_prog.pack(fill="x", padx=10, pady=4)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame_prog, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame_prog, textvariable=self.status_var).pack(anchor="w", pady=(4, 0))

        # --- Log ---
        frame_log = ttk.LabelFrame(self.root, text="Log", padding=8)
        frame_log.pack(fill="both", expand=True, padx=10, pady=4)

        log_controls = ttk.Frame(frame_log)
        log_controls.pack(fill="x", pady=(0, 6))
        ttk.Button(log_controls, text="Copy Output", command=self._copy_output).pack(side="left")

        self.log_text = tk.Text(frame_log, height=8, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)

        # --- Buttons ---
        frame_btns = ttk.Frame(self.root, padding=8)
        frame_btns.pack(fill="x", padx=10, pady=(0, 10))

        self.start_btn = ttk.Button(frame_btns, text="Start Processing", command=self._start)
        self.start_btn.pack(side="left")

        self.cancel_btn = ttk.Button(frame_btns, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(6, 0))

        self.save_close_btn = ttk.Button(frame_btns, text="Save and Close", command=self._save_and_close)
        self.save_close_btn.pack(side="right")

    # --- UI helpers ---

    def _log(self, msg: str):
        self.last_log_was_progress = False
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _log_progress(self, msg: str):
        self.log_text.config(state="normal")
        if self.last_log_was_progress:
            self.log_text.delete("end-2l linestart", "end-1l linestart")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.last_log_was_progress = True

    def _looks_like_progress_line(self, line: str) -> bool:
        if re.match(r"^\s*\d{1,3}%\|", line):
            return True
        return "%|" in line and "|" in line

    def _copy_output(self):
        output = self.log_text.get("1.0", "end-1c")
        if not output.strip():
            messagebox.showinfo("Copy Output", "There is no output to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self.status_var.set("Output copied to clipboard")

    def _browse_folder(self):
        folder = filedialog.askdirectory(initialdir=self.dir_var.get())
        if folder:
            self.dir_var.set(folder)
            self._load_files()

    def _load_files(self):
        self.file_list.delete(0, "end")
        input_dir = Path(self.dir_var.get())
        self.media_files = []
        if not input_dir.is_dir():
            return

        media_files = sorted(
            (f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda p: str(p).lower(),
        )

        self.media_files = media_files
        for media_file in media_files:
            relative_path = media_file.relative_to(input_dir)
            self.file_list.insert("end", str(relative_path))

        if not media_files:
            self._log("No MP3/WEBM/MP4 files found in selected folder.")
        self._select_all()

    def _select_all(self):
        self.file_list.select_set(0, "end")

    def _deselect_all(self):
        self.file_list.select_clear(0, "end")

    def _get_selected_files(self) -> list[Path]:
        indices = self.file_list.curselection()
        return [self.media_files[i] for i in indices]

    def _compute_file_sha256(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _metadata_path(self, output_file: Path) -> Path:
        return output_file.with_suffix(output_file.suffix + ".meta.json")

    def _load_json(self, file_path: Path):
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_saved_state(self):
        secrets = self._load_json(SECRETS_FILE)
        if isinstance(secrets, dict):
            token = str(secrets.get("hf_token", "")).strip()
            if token:
                self.hf_token_var.set(token)

        state = self._load_json(APP_STATE_FILE)
        if not isinstance(state, dict):
            return

        input_dir = state.get("input_dir")
        if isinstance(input_dir, str) and input_dir.strip():
            self.dir_var.set(input_dir)

        self.do_transcription_var.set(bool(state.get("run_transcription", True)))
        self.do_diarization_var.set(bool(state.get("run_diarization", False)))
        self.skip_existing_var.set(bool(state.get("skip_existing", True)))
        self.use_gpu_var.set(bool(state.get("use_gpu", False)))

        whisper_model = state.get("whisper_model")
        if isinstance(whisper_model, str) and whisper_model.strip():
            self.model_var.set(whisper_model)

        whisper_language = state.get("whisper_language")
        if isinstance(whisper_language, str) and whisper_language.strip():
            if whisper_language in WHISPER_LANGUAGE_OPTIONS:
                self.language_var.set(whisper_language)

        diar_model = state.get("diarization_model")
        if isinstance(diar_model, str) and diar_model.strip():
            self.diar_model_var.set(diar_model)

        num_speakers = state.get("num_speakers")
        if isinstance(num_speakers, str):
            self.num_speakers_var.set(num_speakers)

    def _save_state(self):
        state = {
            "input_dir": self.dir_var.get().strip(),
            "run_transcription": self.do_transcription_var.get(),
            "run_diarization": self.do_diarization_var.get(),
            "skip_existing": self.skip_existing_var.get(),
            "whisper_model": self.model_var.get().strip(),
            "whisper_language": self.language_var.get().strip(),
            "diarization_model": self.diar_model_var.get().strip(),
            "use_gpu": self.use_gpu_var.get(),
            "num_speakers": self.num_speakers_var.get().strip(),
        }
        APP_STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        secrets = {
            "hf_token": self.hf_token_var.get().strip(),
        }
        SECRETS_FILE.write_text(
            json.dumps(secrets, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _on_close(self):
        try:
            self._save_state()
        except Exception:
            pass
        self.root.destroy()

    def _save_and_close(self):
        if self.running:
            should_close = messagebox.askyesno(
                "Processing in progress",
                "Processing is still running. Save settings and stop current job before closing?",
            )
            if not should_close:
                return
            self.cancel_requested = True
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()

        try:
            self._save_state()
            self.status_var.set("Settings saved")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save settings:\n{e}")
            return

        self.root.destroy()

    def _is_cached_output_valid(
        self,
        output_file: Path,
        media_file: Path,
        media_hash: str,
        task: str,
        params: dict,
    ) -> bool:
        metadata_file = self._metadata_path(output_file)
        if not output_file.exists() or not metadata_file.exists():
            return False

        metadata = self._load_json(metadata_file)
        if not metadata:
            return False

        return (
            metadata.get("task") == task
            and metadata.get("source_file") == str(media_file.resolve())
            and metadata.get("source_hash_sha256") == media_hash
            and metadata.get("params") == params
            and metadata.get("output_file") == str(output_file.resolve())
        )

    def _write_output_metadata(
        self,
        output_file: Path,
        media_file: Path,
        media_hash: str,
        task: str,
        params: dict,
        progress_log_file: Path,
    ):
        stat = media_file.stat()
        metadata = {
            "task": task,
            "source_file": str(media_file.resolve()),
            "source_hash_sha256": media_hash,
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "params": params,
            "output_file": str(output_file.resolve()),
            "progress_log_file": str(progress_log_file.resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._metadata_path(output_file).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Transcription logic ---

    def _start(self):
        files = self._get_selected_files()
        if not files:
            messagebox.showwarning("No files", "Select at least one MP3, WEBM, or MP4 file.")
            return

        if not self.do_transcription_var.get() and not self.do_diarization_var.get():
            messagebox.showwarning("No task selected", "Enable transcription and/or diarization.")
            return

        if self.do_transcription_var.get() and importlib.util.find_spec("whisper") is None:
            messagebox.showerror("Missing Dependency", "openai-whisper package not installed.\nRun: pip install openai-whisper")
            return

        if self.do_diarization_var.get():
            diarization_script = Path(__file__).with_name("diarization.py")
            if not diarization_script.exists():
                messagebox.showerror("Missing Script", f"Cannot find diarization script:\n{diarization_script}")
                return
            if importlib.util.find_spec("pyannote") is None:
                messagebox.showerror(
                    "Missing Dependency",
                    "pyannote.audio is not installed.\nRun: pip install pyannote.audio",
                )
                return
            if not self.hf_token_var.get().strip():
                messagebox.showerror("HF token required", "Enter a Hugging Face token for diarization.")
                return

        self._save_state()

        self.running = True
        self.cancel_requested = False
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress_var.set(0)

        thread = threading.Thread(target=self._run_processing, args=(files,), daemon=True)
        thread.start()

    def _cancel(self):
        self.cancel_requested = True
        self.cancel_btn.config(state="disabled")
        self.status_var.set("Cancelling…")
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()

    def _run_whisper_cli(
        self,
        media_file: Path,
        output_dir: Path,
        model_name: str,
        language: str,
        progress_log_file: Path,
    ) -> int:
        cmd = [
            sys.executable,
            "-m",
            "whisper",
            str(media_file),
            "--model",
            model_name,
            "--task",
            "transcribe",
            "--output_dir",
            str(output_dir),
            "--output_format",
            "all",
            "--verbose",
            "True",
            "--fp16",
            "False",
        ]
        if language and language.lower() != "auto":
            cmd.extend(["--language", language])
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert self.current_process.stdout is not None
        progress_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_log_file, "a", encoding="utf-8") as log_fp:
            for line in self.current_process.stdout:
                line = line.rstrip("\r\n")
                if line:
                    log_fp.write(line + "\n")
                    if self._looks_like_progress_line(line):
                        self.root.after(0, self._log_progress, line)
                    else:
                        self.root.after(0, self._log, line)

        return_code = self.current_process.wait()
        self.current_process = None
        return return_code

    def _run_merge_speaker_srt_cli(
        self,
        diarization_file: Path,
        transcript_srt_file: Path,
        output_srt_file: Path,
        progress_log_file: Path,
    ) -> int:
        merge_script = Path(__file__).with_name("merge-transcript-diarization.py")
        cmd = [
            sys.executable,
            str(merge_script),
            "--diarization_file",
            str(diarization_file),
            "--srt_file",
            str(transcript_srt_file),
            "--output_file",
            str(output_srt_file),
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert self.current_process.stdout is not None
        progress_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_log_file, "a", encoding="utf-8") as log_fp:
            for line in self.current_process.stdout:
                line = line.rstrip("\r\n")
                if line:
                    log_fp.write(line + "\n")
                    if self._looks_like_progress_line(line):
                        self.root.after(0, self._log_progress, line)
                    else:
                        self.root.after(0, self._log, line)

        return_code = self.current_process.wait()
        self.current_process = None
        return return_code

    def _run_diarization_cli(
        self,
        media_file: Path,
        output_file: Path,
        model_name: str,
        token: str,
        use_gpu: bool,
        num_speakers: str,
        progress_log_file: Path,
    ) -> int:
        diarization_script = Path(__file__).with_name("diarization.py")
        cmd = [
            sys.executable,
            str(diarization_script),
            "--token",
            token,
            "--model",
            model_name,
            "--audio_file",
            str(media_file),
            "--diarization_output",
            str(output_file),
        ]
        if use_gpu:
            cmd.append("--use_gpu")
        if num_speakers.strip():
            cmd.extend(["--num_speakers", num_speakers.strip()])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert self.current_process.stdout is not None
        progress_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_log_file, "a", encoding="utf-8") as log_fp:
            for line in self.current_process.stdout:
                line = line.rstrip("\r\n")
                if line:
                    log_fp.write(line + "\n")
                    if self._looks_like_progress_line(line):
                        self.root.after(0, self._log_progress, line)
                    else:
                        self.root.after(0, self._log, line)

        return_code = self.current_process.wait()
        self.current_process = None
        return return_code

    def _run_processing(self, files: list[Path]):
        run_transcription = self.do_transcription_var.get()
        run_diarization = self.do_diarization_var.get()
        skip_existing = self.skip_existing_var.get()
        whisper_model_name = self.model_var.get()
        whisper_language = self.language_var.get().strip().lower() or "auto"
        diarization_model_name = self.diar_model_var.get().strip() or DEFAULT_DIARIZATION_MODEL
        hf_token = self.hf_token_var.get().strip()
        num_speakers = self.num_speakers_var.get().strip()
        use_gpu = self.use_gpu_var.get()
        input_root = Path(self.dir_var.get())

        transcript_dir = input_root / "transcripts"
        diarization_dir = input_root / "diarization"
        if run_transcription:
            transcript_dir.mkdir(exist_ok=True)
            self.root.after(0, self._log, f"Using Whisper model '{whisper_model_name}'.")
            if whisper_language == "auto":
                self.root.after(0, self._log, "Whisper language: auto-detect")
            else:
                self.root.after(0, self._log, f"Whisper language: {whisper_language}")
        if run_diarization:
            diarization_dir.mkdir(exist_ok=True)
            self.root.after(0, self._log, f"Using diarization model '{diarization_model_name}'.")

        total = len(files)
        tasks = []
        if run_transcription:
            tasks.append("transcription")
        if run_diarization:
            tasks.append("diarization + merge")
        self.root.after(0, self._log, f"=== Starting: {', '.join(tasks)} for {total} file(s) ===")
        pipeline_t0 = time.time()

        for i, media_file in enumerate(files):
            if self.cancel_requested:
                self.root.after(0, self._log, "Cancelled by user.")
                break

            file_t0 = time.time()
            file_size_mb = media_file.stat().st_size / (1024 * 1024)
            self.root.after(0, self.status_var.set, f"Processing {i + 1}/{total}: {media_file.name}")
            self.root.after(0, self._log, f"\n--- [{i + 1}/{total}] {media_file.name} ({file_size_mb:.1f} MB) ---")

            self.root.after(0, self._log, "  Computing file hash…")
            try:
                hash_t0 = time.time()
                media_hash = self._compute_file_sha256(media_file)
                hash_dt = time.time() - hash_t0
                self.root.after(0, self._log, f"  Hash computed in {hash_dt:.1f}s")
            except Exception as e:
                self.root.after(0, self._log, f"  FAILED hashing: {e}")
                self.root.after(0, self.progress_var.set, (i + 1) / total * 100)
                continue

            relative_media = media_file.relative_to(input_root)

            if run_transcription:
                txt_file = (transcript_dir / relative_media).with_suffix(".txt")
                srt_file = (transcript_dir / relative_media).with_suffix(".srt")
                txt_file.parent.mkdir(parents=True, exist_ok=True)
                transcription_params = {
                    "model": whisper_model_name,
                    "language": whisper_language,
                }
                transcript_progress_log = txt_file.with_suffix(".transcription.progress.log")

                self.root.after(0, self._log, "  [Transcription] Checking cache…")
                if self._is_cached_output_valid(
                    txt_file,
                    media_file,
                    media_hash,
                    "transcription",
                    transcription_params,
                ) and srt_file.exists():
                    self.root.after(0, self._log, f"  [Transcription] Cache valid – skipping ({txt_file.name})")
                elif skip_existing and txt_file.exists():
                    self.root.after(0, self._log, f"  [Transcription] Output exists (no valid cache) – skipping")
                else:
                    self.root.after(0, self._log, f"  [Transcription] Running Whisper (model={whisper_model_name}, lang={whisper_language})…")
                    try:
                        whisper_t0 = time.time()
                        return_code = self._run_whisper_cli(
                            media_file,
                            txt_file.parent,
                            whisper_model_name,
                            whisper_language,
                            transcript_progress_log,
                        )
                        whisper_dt = time.time() - whisper_t0
                        if return_code == 0:
                            self.root.after(0, self._log, f"  [Transcription] Done in {self._fmt_duration(whisper_dt)} -> {txt_file.name}")
                            self._write_output_metadata(
                                txt_file,
                                media_file,
                                media_hash,
                                "transcription",
                                transcription_params,
                                transcript_progress_log,
                            )
                        elif self.cancel_requested:
                            self.root.after(0, self._log, "  [Transcription] Cancelled.")
                            break
                        else:
                            self.root.after(0, self._log, f"  [Transcription] FAILED (exit code {return_code}) after {self._fmt_duration(whisper_dt)}")
                    except Exception as e:
                        self.root.after(0, self._log, f"  [Transcription] FAILED: {e}")

            if self.cancel_requested:
                break

            if run_diarization:
                diar_base = (diarization_dir / relative_media).with_suffix("")
                diar_file = diar_base.parent / f"{diar_base.name}_diarization.txt"
                speaker_srt_file = diar_base.parent / f"{diar_base.name}_speaker.srt"
                diar_file.parent.mkdir(parents=True, exist_ok=True)
                diarization_params = {
                    "model": diarization_model_name,
                    "use_gpu": use_gpu,
                    "num_speakers": num_speakers or None,
                }
                diarization_progress_log = diar_file.with_suffix(".diarization.progress.log")

                self.root.after(0, self._log, "  [Diarization] Checking cache…")
                if self._is_cached_output_valid(
                    diar_file,
                    media_file,
                    media_hash,
                    "diarization",
                    diarization_params,
                ) and speaker_srt_file.exists():
                    self.root.after(0, self._log, f"  [Diarization] Cache valid – skipping ({diar_file.name})")
                elif skip_existing and diar_file.exists():
                    self.root.after(0, self._log, f"  [Diarization] Output exists (no valid cache) – skipping")
                else:
                    gpu_label = "GPU" if use_gpu else "CPU"
                    spk_label = f", speakers={num_speakers}" if num_speakers else ""
                    self.root.after(0, self._log, f"  [Diarization] Running pyannote ({gpu_label}{spk_label})…")
                    try:
                        diar_t0 = time.time()
                        return_code = self._run_diarization_cli(
                            media_file,
                            diar_file,
                            diarization_model_name,
                            hf_token,
                            use_gpu,
                            num_speakers,
                            diarization_progress_log,
                        )
                        diar_dt = time.time() - diar_t0
                        if return_code == 0:
                            self.root.after(0, self._log, f"  [Diarization] Done in {self._fmt_duration(diar_dt)} -> {diar_file.name}")
                            self._write_output_metadata(
                                diar_file,
                                media_file,
                                media_hash,
                                "diarization",
                                diarization_params,
                                diarization_progress_log,
                            )

                            transcript_srt_file = (transcript_dir / relative_media).with_suffix(".srt")
                            merge_progress_log = speaker_srt_file.with_suffix(".merge.progress.log")
                            if transcript_srt_file.exists():
                                self.root.after(0, self._log, f"  [Merge] Combining diarization + transcript SRT…")
                                merge_t0 = time.time()
                                merge_code = self._run_merge_speaker_srt_cli(
                                    diar_file,
                                    transcript_srt_file,
                                    speaker_srt_file,
                                    merge_progress_log,
                                )
                                merge_dt = time.time() - merge_t0
                                if merge_code == 0:
                                    self.root.after(0, self._log, f"  [Merge] Done in {self._fmt_duration(merge_dt)} -> {speaker_srt_file.name}")
                                else:
                                    self.root.after(0, self._log, f"  [Merge] FAILED (exit code {merge_code})")
                            else:
                                self.root.after(
                                    0,
                                    self._log,
                                    f"  [Merge] WARNING: cannot create speaker SRT, missing transcript SRT ({transcript_srt_file.name})",
                                )
                        elif self.cancel_requested:
                            self.root.after(0, self._log, "  [Diarization] Cancelled.")
                            break
                        else:
                            self.root.after(0, self._log, f"  [Diarization] FAILED (exit code {return_code}) after {self._fmt_duration(diar_dt)}")
                    except Exception as e:
                        self.root.after(0, self._log, f"  [Diarization] FAILED: {e}")

            file_dt = time.time() - file_t0
            self.root.after(0, self._log, f"  File completed in {self._fmt_duration(file_dt)}")
            self.root.after(0, self.progress_var.set, (i + 1) / total * 100)

        pipeline_dt = time.time() - pipeline_t0
        self.root.after(0, self._finish, pipeline_dt)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def _finish(self, elapsed: float = 0):
        self.running = False
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if self.cancel_requested:
            self.status_var.set("Cancelled")
            self._log(f"\n=== Cancelled after {self._fmt_duration(elapsed)}. ===")
        else:
            self.status_var.set("Done")
            self._log(f"\n=== All done in {self._fmt_duration(elapsed)}. ===")


def main():
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
