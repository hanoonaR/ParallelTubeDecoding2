# Preparing PTD training data

The preparation scripts convert VidSTG and HC-STVG annotations into the JSON
consumed by PTD SFT. Run them from the repository root.

Videos are sampled at 2 FPS with at most 64 frames. Every sampled frame has one
time token, and boxes are converted to integer coordinates from 0 to 1000.

## Directory layout

The examples below use one common video root:

```text
VIDEO_ROOT/
├── vidstg/video/
├── hc-stvg_v1/videos_v1/
└── hc-stvg_v2/v2_videos_train/video_parts/
```

`--video-root` is the location used while preparing the annotations.
`--video-prefix` is the path stored in the output JSON and is resolved under
`VIDEO_ROOT` during training.

## VidSTG

The VidSTG input JSON contains top-level `videos` and `trajectories` fields.

```bash
python data/prepare_vidstg.py \
  --annotations /path/to/vidstg_annotations/train.json \
  --video-root /path/to/VIDEO_ROOT/vidstg/video \
  --video-prefix vidstg/video \
  --output /path/to/ptd_sft.json
```

## HC-STVG v1

HC-STVG v1 stores videos directly under `videos_v1`. Its tube end is derived
from `st_frame` and the number of boxes in the trajectory.

```bash
python data/prepare_hcstvg_v1.py \
  --annotations /path/to/hc-stvg_v1/train.json \
  --video-root /path/to/VIDEO_ROOT/hc-stvg_v1/videos_v1 \
  --video-prefix hc-stvg_v1/videos_v1 \
  --output /path/to/ptd_sft.json \
  --append
```

## HC-STVG v2

HC-STVG v2 provides `ed_frame` and may distribute videos across part
directories. Pass `video_parts.json` when that mapping is used.

```bash
python data/prepare_hcstvg_v2.py \
  --annotations /path/to/hc-stvg_v2/annos/train.json \
  --video-root /path/to/VIDEO_ROOT/hc-stvg_v2/v2_videos_train/video_parts \
  --video-parts /path/to/hc-stvg_v2/video_parts.json \
  --video-prefix hc-stvg_v2/v2_videos_train/video_parts \
  --output /path/to/ptd_sft.json \
  --append
```

If the v2 videos are in one directory, omit `--video-parts`.

The first command creates the output JSON. The two `--append` commands add the
HC-STVG samples to the same list. Use `--limit` with any script for a small
data-loading check.

## Output format

Each sample contains one video and one user/assistant pair:

```json
{
  "video": "vidstg/video/example.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nGiven the query: 'A person opens the door.' Localize the described object throughout the video. Use object reference tokens, time tokens, and box tokens. Return the object reference, event time segment, and per-time bbox coordinates."
    },
    {
      "from": "gpt",
      "value": "<|object_ref_start|>A person opens the door<|object_ref_end|>\n<|time_start|><t3><t4><|time_end|>\n<t3><|box_start|><100><120><420><900><|box_end|>\n<t4><|box_start|><110><125><430><900><|box_end|>"
    }
  ]
}
```

The prepared annotation format is accepted by the SFT data loader.
