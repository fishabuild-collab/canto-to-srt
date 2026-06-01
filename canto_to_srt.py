#!/usr/bin/env python3
"""
Canto-to-SRT: Transcribe Cantonese media files to SRT subtitles using Whisper.
Usage:
    python canto_to_srt.py [input_folder] [output_folder] [--model MODEL] [--language LANG]

Defaults:
    input_folder  = ./input
    output_folder = ./output
    model         = medium  (tiny | base | small | medium | large)
    language      = zh      (Chinese; covers Cantonese — 'yue' may crash on some model weights)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import whisper


def get_device() -> tuple[str, bool]:
    """Return (device, use_fp16). MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps", True
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus",
}


def seconds_to_srt_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seconds_to_srt_timestamp(seg["start"])
        end = seconds_to_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def find_media_files(folder: Path) -> list[Path]:
    files = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def transcribe_file(model, media_path: Path, language: str, fp16: bool) -> list[dict]:
    try:
        result = model.transcribe(str(media_path), language=language, fp16=fp16, verbose=False)
    except (ValueError, IndexError):
        # 'yue' token missing from some model weights — retry with auto-detect
        print(f"  Warning: language '{language}' not supported by this model; retrying with auto-detect.")
        result = model.transcribe(str(media_path), fp16=fp16, verbose=False)
    return result["segments"]


def process_folder(input_folder: Path, output_folder: Path, model_name: str, language: str):
    output_folder.mkdir(parents=True, exist_ok=True)

    media_files = find_media_files(input_folder)
    if not media_files:
        print(f"No supported media files found in: {input_folder}")
        return

    device, fp16 = get_device()
    print(f"Loading Whisper model '{model_name}' on device: {device}")
    model = whisper.load_model(model_name, device=device)
    print(f"Model loaded. Found {len(media_files)} file(s) to process.\n")

    succeeded, failed = 0, 0

    for i, media_path in enumerate(media_files, start=1):
        rel = media_path.relative_to(input_folder)
        srt_path = output_folder / rel.with_suffix(".srt")
        srt_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(media_files)}] {rel}")
        start_time = time.time()

        try:
            segments = transcribe_file(model, media_path, language, fp16)
            srt_content = segments_to_srt(segments)
            srt_path.write_text(srt_content, encoding="utf-8")
            elapsed = time.time() - start_time
            print(f"  -> {srt_path.name}  ({elapsed:.1f}s)\n")
            succeeded += 1
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            failed += 1

    print(f"Done. {succeeded} succeeded, {failed} failed.")
    print(f"SRT files saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe Cantonese media to SRT")
    parser.add_argument("input_folder", nargs="?", default="input",
                        help="Folder containing media files (default: ./input)")
    parser.add_argument("output_folder", nargs="?", default="output",
                        help="Folder to write SRT files (default: ./output)")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", default="zh",
                        help="Language code: zh=Chinese (Cantonese/Mandarin), yue=Cantonese if supported (default: zh)")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not input_folder.exists():
        print(f"Error: input folder does not exist: {input_folder}")
        sys.exit(1)

    process_folder(input_folder, output_folder, args.model, args.language)


if __name__ == "__main__":
    main()
