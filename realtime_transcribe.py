"""
Real-time microphone transcription using local Whisper model.

Usage:
    python realtime_transcribe.py [--model base] [--language it] [--chunk 5] [--device 0]

Options:
    --model     Whisper model size: tiny, base, small, medium, large (default: base)
    --language  Language code or 'auto' for auto-detection (default: auto)
    --chunk     Seconds of audio per transcription chunk (default: 5)
    --device    Audio input device index (default: system default)

Press Ctrl+C to stop.
"""

import argparse
import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import whisper

WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz mono audio


def get_device_sample_rate(device_index):
    """Get the default sample rate for a device, fallback to 16000."""
    if device_index is not None:
        info = sd.query_devices(device_index)
    else:
        info = sd.query_devices(sd.default.device[0])
    return int(info["default_samplerate"])


def resample(audio, orig_sr, target_sr):
    """Simple linear interpolation resampling."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time microphone transcription")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default="auto",
                        help="Language code (e.g., 'it', 'en') or 'auto' (default: auto)")
    parser.add_argument("--chunk", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index (default: system default)")
    parser.add_argument("--energy-threshold", type=float, default=0.01,
                        help="Minimum RMS energy to trigger transcription (default: 0.01)")
    return parser.parse_args()


def list_devices():
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']} (channels: {d['max_input_channels']}){marker}")
    print()


def main():
    args = parse_args()

    list_devices()
    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)
    print(f"Model loaded. Listening... (chunk={args.chunk}s, language={args.language})")
    print("=" * 60)
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    audio_queue = queue.Queue()
    device_sr = get_device_sample_rate(args.device)
    print(f"Device sample rate: {device_sr} Hz")
    chunk_samples = int(args.chunk * device_sr)  # samples at device rate

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio warning: {status}]", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    try:
        stream = sd.InputStream(
            samplerate=device_sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=args.device,
            callback=audio_callback,
        )
        stream.start()
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        print("Try specifying a device with --device <index>")
        sys.exit(1)

    transcript_lines = []

    try:
        buffer = np.empty(0, dtype=np.float32)

        while True:
            # Collect audio until we have a full chunk
            while len(buffer) < chunk_samples:
                data = audio_queue.get()
                buffer = np.concatenate([buffer, data])

            # Extract the chunk and resample to 16kHz for Whisper
            chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            # Resample to Whisper's expected 16kHz
            chunk_16k = resample(chunk, device_sr, WHISPER_SAMPLE_RATE)

            # Skip silent chunks
            rms = np.sqrt(np.mean(chunk_16k ** 2))
            if rms < args.energy_threshold:
                continue

            # Transcribe
            options = {"fp16": False}
            if args.language.lower() != "auto":
                options["language"] = args.language

            result = model.transcribe(chunk_16k, **options)
            text = result["text"].strip()

            if text:
                transcript_lines.append(text)
                print(f">> {text}")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopped.")
        stream.stop()
        stream.close()

        if transcript_lines:
            print("\n--- Full Transcript ---")
            full_text = "\n".join(transcript_lines)
            print(full_text)

            # Save to file
            output_file = "realtime_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
