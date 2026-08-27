#!/usr/bin/env python3
"""Prepare HC-STVG v1 annotations for PTD SFT."""

import argparse
import json
import math
from pathlib import Path

import torch


PROMPT = (
    "Localize the described object throughout the video. Use object reference "
    "tokens, time tokens, and box tokens. Return the object reference, event "
    "time segment, and per-time bbox coordinates."
)


def read_video(path):
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    frame_count = len(reader)
    fps = float(reader.get_avg_fps())
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return frame_count, fps


def sample_frames(frame_count, fps):
    count = math.floor(frame_count / fps * 2.0)
    count = min(max(count, 4), 64, frame_count)
    return torch.linspace(0, frame_count - 1, count).round().long().tolist()


def normalize_box(box, width, height):
    x, y, box_width, box_height = [float(value) for value in box]
    x1, x2 = max(0.0, x), min(width, x + box_width)
    y1, y2 = max(0.0, y), min(height, y + box_height)
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        return None
    normalized = [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return None
    return normalized


def build_record(video, caption, boxes):
    if not boxes:
        return None
    indices = [item[0] for item in boxes]
    if indices != list(range(indices[0], indices[-1] + 1)):
        return None
    query = caption if caption.endswith((".", "?", "!")) else f"{caption}."
    answer = [f"<|object_ref_start|>{caption}<|object_ref_end|>"]
    answer.append(f"<|time_start|><t{indices[0]}><t{indices[-1]}><|time_end|>")
    for time_index, box in boxes:
        coordinates = "".join(f"<{value}>" for value in box)
        answer.append(f"<t{time_index}><|box_start|>{coordinates}<|box_end|>")
    return {
        "video": video,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\nGiven the query: '{query}' {PROMPT}",
            },
            {"from": "gpt", "value": "\n".join(answer)},
        ],
    }


def prepare(args):
    with open(args.annotations) as handle:
        annotations = json.load(handle)
    records = []
    skipped = 0
    for video_name, sample in annotations.items():
        if args.limit and len(records) >= args.limit:
            break
        try:
            frame_count, fps = read_video(Path(args.video_root) / video_name)
        except (FileNotFoundError, RuntimeError, ValueError):
            skipped += 1
            continue

        trajectory = sample.get("bbox", [])
        caption = str(sample.get("caption") or "").strip()
        annotation_frames = int(sample["img_num"])
        tube_start = int(sample["st_frame"])
        tube_end = tube_start + len(trajectory) - 1
        if not trajectory or not caption or annotation_frames <= 0:
            skipped += 1
            continue

        boxes = []
        for time_index, video_frame in enumerate(sample_frames(frame_count, fps), start=1):
            annotation_frame = (
                1
                if annotation_frames == 1 or frame_count == 1
                else round(video_frame * (annotation_frames - 1) / (frame_count - 1)) + 1
            )
            if not tube_start <= annotation_frame <= tube_end:
                continue
            box = normalize_box(
                trajectory[annotation_frame - tube_start],
                float(sample["width"]),
                float(sample["height"]),
            )
            if box is not None:
                boxes.append((time_index, box))

        record = build_record(
            str(Path(args.video_prefix) / video_name), caption, boxes
        )
        if record is None:
            skipped += 1
        else:
            records.append(record)
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--video-prefix", default="hc-stvg_v1/videos_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records, skipped = prepare(args)
    output_path = Path(args.output)
    if args.append and output_path.exists():
        with output_path.open() as handle:
            records = json.load(handle) + records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} samples to {output_path}; skipped {skipped}")


if __name__ == "__main__":
    main()
