#!/usr/bin/env python3
"""
Speaker Diarization Script using PyAnnote

This script performs speaker diarization on an audio file using a pre-trained
model from Hugging Face. It identifies different speakers in the audio and
outputs their speaking time segments.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Speaker Diarization using PyAnnote")
    parser.add_argument(
        "--token", type=str, required=True, help="HuggingFace token for authentication"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Model name to use for diarization (default: pyannote/speaker-diarization-3.1)",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing")
    parser.add_argument(
        "--audio_file", type=str, required=True, help="Path to the audio file"
    )
    parser.add_argument(
        "--num_speakers", type=int, default=None, help="Number of speakers (optional)"
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (optional)",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (optional)",
    )
    parser.add_argument(
        "--diarization_output",
        type=str,
        required=True,
        help="Path to save the diarization output",
    )
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    if args.num_speakers is not None:
        args.min_speakers = args.num_speakers
        args.max_speakers = args.num_speakers

    if args.min_speakers is not None and args.max_speakers is not None:
        if args.min_speakers > args.max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers")


def print_configuration(args):
    """Print the diarization configuration."""
    print(f"Model: {args.model}")
    if args.num_speakers is not None:
        print(f"Number of speakers: {args.num_speakers}")
    elif args.min_speakers is not None and args.max_speakers is not None:
        print(
            f"Number of speakers will be estimated within the range: "
            f"[{args.min_speakers}, {args.max_speakers}]"
        )
    elif args.min_speakers is not None:
        print(f"Minimum number of speakers: {args.min_speakers}")
    elif args.max_speakers is not None:
        print(f"Maximum number of speakers: {args.max_speakers}")


def load_pipeline(model_name, token, use_gpu):
    """Load the diarization pipeline."""
    print(f"Loading pipeline '{model_name}'…")
    t0 = time.time()
    # Support both old and new pyannote signatures.
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)

    if use_gpu:
        print("Moving pipeline to GPU…")
        pipeline.to(torch.device("cuda"))

    dt = time.time() - t0
    print(f"Pipeline loaded in {dt:.1f}s")
    return pipeline


def run_diarization(pipeline, audio_file, num_speakers, min_speakers, max_speakers):
    """Run the diarization pipeline on the audio file."""
    print("Running diarization…")
    t0 = time.time()
    with ProgressHook() as hook:
        result = pipeline(
            audio_file,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook,
        )
    dt = time.time() - t0
    print(f"Diarization finished in {dt:.1f}s")
    return result


def prepare_audio_for_diarization(audio_file: str):
    """Convert unsupported formats to a temporary WAV file for pyannote."""
    source = Path(audio_file)
    if source.suffix.lower() == ".wav":
        return str(source), None

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to diarize non-WAV files. Install ffmpeg and add it to PATH."
        )

    tmp = tempfile.NamedTemporaryFile(prefix="diarization_", suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(tmp_path),
    ]
    print(f"Converting input for diarization: {source.name} -> WAV (16kHz mono)…")
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"Audio conversion done in {dt:.1f}s")
    return str(tmp_path), str(tmp_path)


def save_results(output, output_path):
    """Save diarization results to a file."""
    annotation = getattr(output, "speaker_diarization", output)
    segments = list(annotation.itertracks(yield_label=True))
    speakers = set(s for _, _, s in segments)
    print(f"Detected {len(speakers)} speaker(s), {len(segments)} segment(s)")
    print(f"Saving results to {output_path}…")
    with open(output_path, "w") as fp:
        for turn, _, speaker in segments:
            fp.write(f"{speaker} start={turn.start:.2f}s stop={turn.end:.2f}s\n")
    print("Results saved.")


def main():
    """Main function."""
    args = parse_arguments()
    validate_arguments(args)
    print_configuration(args)

    pipeline = load_pipeline(args.model, args.token, args.use_gpu)
    prepared_audio, temp_audio = prepare_audio_for_diarization(args.audio_file)
    try:
        output = run_diarization(
            pipeline,
            prepared_audio,
            args.num_speakers,
            args.min_speakers,
            args.max_speakers,
        )
        save_results(output, args.diarization_output)
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)


if __name__ == "__main__":
    main()
