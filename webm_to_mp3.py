from pathlib import Path
import subprocess
import sys

# Folder containing the .webm files
INPUT_DIR = Path(r".")  # current folder
OUTPUT_DIR = INPUT_DIR / "mp3"

def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        print("Error: ffmpeg is not installed or not in PATH.")
        sys.exit(1)

def convert_webm_to_mp3(input_file: Path, output_file: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-c:a", "libmp3lame",
        "-q:a", "2",
        str(output_file)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Converted: {input_file.name} -> {output_file.name}")
    except subprocess.CalledProcessError:
        print(f"Failed: {input_file.name}")

def main():
    check_ffmpeg()
    OUTPUT_DIR.mkdir(exist_ok=True)

    webm_files = sorted(INPUT_DIR.glob("*.webm"))

    if not webm_files:
        print("No .webm files found.")
        return

    for webm_file in webm_files:
        mp3_file = OUTPUT_DIR / f"{webm_file.stem}.mp3"
        convert_webm_to_mp3(webm_file, mp3_file)

    print("Done.")

if __name__ == "__main__":
    main()