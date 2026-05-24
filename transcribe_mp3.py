from pathlib import Path
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Folder containing the .mp3 files
INPUT_DIR = Path(r".") / "mp3"
OUTPUT_DIR = INPUT_DIR / "transcripts"


def transcribe_mp3(client: OpenAI, input_file: Path, output_file: Path):
    try:
        with open(input_file, "rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
            )
        output_file.write_text(result.text, encoding="utf-8")
        print(f"Transcribed: {input_file.name} -> {output_file.name}")
    except Exception as e:
        print(f"Failed: {input_file.name} — {e}")


def main():
    client = OpenAI()  # uses OPENAI_API_KEY env var
    OUTPUT_DIR.mkdir(exist_ok=True)

    mp3_files = sorted(INPUT_DIR.glob("*.mp3"))

    if not mp3_files:
        print("No .mp3 files found.")
        return

    for mp3_file in mp3_files:
        txt_file = OUTPUT_DIR / f"{mp3_file.stem}.txt"
        if txt_file.exists():
            print(f"Skipped (already exists): {txt_file.name}")
            continue
        transcribe_mp3(client, mp3_file, txt_file)

    print("Done.")


if __name__ == "__main__":
    main()
