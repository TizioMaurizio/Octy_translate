import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass
class SubtitleSegment:
    index: str
    start: float
    end: float
    text: str


@dataclass
class MergedSegment:
    index: int
    start: float
    end: float
    speaker: str
    text: str


def load_diarization(diarization_file: str) -> List[DiarizationSegment]:
    """Load diarization data from file."""
    diarization = []
    with open(diarization_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[1].split("=")[1][:-1])
            end = float(parts[2].split("=")[1][:-1])
            speaker = parts[0]
            diarization.append(DiarizationSegment(start, end, speaker))
    return diarization


def load_srt(srt_file: str) -> List[SubtitleSegment]:
    """Load SRT subtitle data from file."""
    subtitles = []
    with open(srt_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip().split("\n\n")
        for block in content:
            lines = block.split("\n")
            if len(lines) < 3:
                continue

            index = lines[0]
            time_range = lines[1]
            text = " ".join(lines[2:])

            start_str, end_str = time_range.split(" --> ")
            start = srt_time_to_seconds(start_str)
            end = srt_time_to_seconds(end_str)

            subtitles.append(SubtitleSegment(index, start, end, text))
    return subtitles


def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds."""
    h, m, s = time_str.split(":")
    s, ms = s.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format."""
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def find_best_speaker_for_subtitle(
    subtitle: SubtitleSegment, diarization: List[DiarizationSegment]
) -> str:
    """Find the speaker with maximum overlap for a given subtitle."""
    best_speaker = "UNKNOWN"
    max_overlap = 0

    for segment in diarization:
        overlap = calculate_overlap(
            subtitle.start, subtitle.end, segment.start, segment.end
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = segment.speaker

    return best_speaker


def merge_diarization_srt(
    diarization: List[DiarizationSegment], subtitles: List[SubtitleSegment]
) -> List[MergedSegment]:
    """Merge diarization data with subtitles."""
    merged = []
    for i, subtitle in enumerate(subtitles, start=1):
        best_speaker = find_best_speaker_for_subtitle(subtitle, diarization)
        merged.append(
            MergedSegment(
                i,
                subtitle.start,
                subtitle.end,
                best_speaker,
                subtitle.text,
            )
        )
    return merged


def aggregate_speaker_segments(
    segments: List[MergedSegment], max_gap_seconds: float = 0.8
) -> List[MergedSegment]:
    """Aggregate adjacent segments from the same speaker into longer phrases."""
    if not segments:
        return []

    aggregated: List[MergedSegment] = []
    current = MergedSegment(
        index=1,
        start=segments[0].start,
        end=segments[0].end,
        speaker=segments[0].speaker,
        text=segments[0].text.strip(),
    )

    for segment in segments[1:]:
        gap = segment.start - current.end
        can_merge = segment.speaker == current.speaker and gap <= max_gap_seconds

        if can_merge:
            if segment.text.strip():
                current.text = (current.text + " " + segment.text.strip()).strip()
            current.end = segment.end
        else:
            aggregated.append(current)
            current = MergedSegment(
                index=len(aggregated) + 1,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                text=segment.text.strip(),
            )

    aggregated.append(current)
    return aggregated


def save_merged_srt(merged: List[MergedSegment], output_file: str) -> None:
    """Save merged data to SRT file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in merged:
            f.write(f"{segment.index}\n")
            f.write(
                f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}\n"
            )
            f.write(f"[{segment.speaker}] {segment.text}\n\n")


def main() -> None:
    """Main function to parse arguments and run the merging process."""
    parser = argparse.ArgumentParser(
        description="Merge Speaker Diarization with SRT Subtitles"
    )
    parser.add_argument(
        "--diarization_file",
        type=str,
        required=True,
        help="Path to the diarization file",
    )
    parser.add_argument(
        "--srt_file", type=str, required=True, help="Path to the SRT subtitle file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the merged SRT file",
    )

    args = parser.parse_args()

    diarization = load_diarization(args.diarization_file)
    subtitles = load_srt(args.srt_file)
    merged = merge_diarization_srt(diarization, subtitles)
    aggregated = aggregate_speaker_segments(merged)
    save_merged_srt(aggregated, args.output_file)

    print(f"Merged diarization and subtitles saved to {args.output_file}")


if __name__ == "__main__":
    main()
