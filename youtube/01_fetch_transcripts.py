#!/usr/bin/env python3
"""
01_fetch_transcripts.py
=======================
Step 1 of the YouTube RAG pipeline.

Reads the playlist metadata from  ../genai_playlist.jsonl
Fetches the English transcript for each video via youtube-transcript-api
Saves the results to  youtube/transcripts.json

Run once (or re-run to refresh):
    conda activate bluParrot
    python youtube/01_fetch_transcripts.py
"""

import json
import time
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

THIS_DIR      = Path(__file__).parent                      # youtube/
PROJECT_DIR   = THIS_DIR.parent                            # Smart Query Assistant/
PLAYLIST_FILE = PROJECT_DIR / "genai_playlist.jsonl"
OUTPUT_FILE   = THIS_DIR    / "transcripts.json"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_playlist(path: Path) -> list[dict]:
    """
    Parse the yt-dlp flat-playlist JSONL file.
    Returns a list of video dicts with only the fields we care about.
    """
    videos = []
    with open(path, "r") as f:
        for line in f:
            raw = json.loads(line.strip())
            # Extract channel name from title (all videos end with "| CampusX")
            title = raw.get("title", "")
            channel = title.split("|")[-1].strip() if "|" in title else "Unknown"

            videos.append({
                "video_id":    raw["id"],
                "title":       title,
                "url":         raw.get("webpage_url", f"https://www.youtube.com/watch?v={raw['id']}"),
                "duration_min": round((raw.get("duration") or 0) / 60),
                "channel":     channel,
            })
    return videos


def get_transcript(video_id: str) -> tuple[str | None, str | None]:
    """
    Fetch and join the full transcript for a video.

    Strategy (v1.x API — instance-based):
      1. Try English first.
      2. Fall back to the first available language (handles Hindi, etc.).

    Returns (transcript_text, language_code) or (None, None) on failure.
    LaBSE embeddings handle multilingual content, so Hindi transcripts with
    English technical terms are fully usable for cross-lingual retrieval.
    """
    api = YouTubeTranscriptApi()
    try:
        # Try English; CampusX videos are in Hindi so we fall back
        try:
            fetched = api.fetch(video_id, languages=["en"])
            lang    = "en"
        except NoTranscriptFound:
            transcript_list = api.list(video_id)
            first_available = next(iter(transcript_list))
            fetched = first_available.fetch()
            lang    = first_available.language_code

        text = " ".join(snippet.text for snippet in fetched.snippets)
        return text, lang

    except (TranscriptsDisabled, VideoUnavailable):
        return None, None
    except Exception as exc:
        print(f"    ⚠  Unexpected error: {exc}")
        return None, None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading playlist from: {PLAYLIST_FILE}")
    videos = parse_playlist(PLAYLIST_FILE)
    print(f"Found {len(videos)} videos in playlist.\n")

    results  = []
    skipped  = 0

    for i, video in enumerate(videos, 1):
        vid_id = video["video_id"]
        title  = video["title"][:60]
        print(f"[{i:2}/{len(videos)}] {vid_id}  {title} …", end=" ", flush=True)

        transcript, lang = get_transcript(vid_id)

        if transcript is None:
            print("✗  no transcript — skipped")
            skipped += 1
            continue

        video["transcript"]      = transcript
        video["transcript_lang"] = lang          # stored in metadata for reference
        results.append(video)

        word_count = len(transcript.split())
        print(f"✓  {word_count:,} words  [{lang}]")

        # Be polite to YouTube — small delay between requests
        time.sleep(0.4)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*55}")
    print(f"  Saved   : {len(results)} videos  →  {OUTPUT_FILE}")
    print(f"  Skipped : {skipped} videos (no transcript available)")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
