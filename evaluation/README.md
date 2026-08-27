# PTD tasks for lmms-eval

The files under `lmms_eval/` mirror their destination in an lmms-eval
checkout. Copy the four task folders and model adapter:

```bash
export PTD_REPO="/path/to/ParallelTubeDecoding"
export LMMS_EVAL_REPO="/path/to/lmms-eval"

cp -r \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/vidstg" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/hcstvg" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/charades_temp_loc" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/activitney_temp_loc" \
  "$LMMS_EVAL_REPO/lmms_eval/tasks/"
cp "$PTD_REPO/evaluation/lmms_eval/lmms_eval/models/simple/ptd_qwen3_vl.py" \
  "$LMMS_EVAL_REPO/lmms_eval/models/simple/"
```

Add the following entry to `AVAILABLE_SIMPLE_MODELS` in
`lmms_eval/models/__init__.py`:

```python
"ptd_qwen3_vl": "PTDQwen3VL",
```

Edit the `data_files.test` path in each PTD task YAML. Relative video paths are
resolved with these variables:

```bash
export VIDSTG_VIDEO_ROOT="/path/to/VidSTG"
export HCSTVG_VIDEO_ROOT="/path/to/HC-STVG"
export CHARADES_STA_VIDEO_ROOT="/path/to/Charades_v1_480"
export ACTIVITYNET_VIDEO_ROOT="/path/to/ActivityNet/videos"
```

VidSTG and HC-STVG annotations are JSONL files with these fields:

- `video_path`, `caption`, and `gt_sampled_frame_boxes`
- `qtype` (`declarative` or `interrogative`) for VidSTG
- `version` (`v1` or `v2`) for HC-STVG
- optional `video_start_sec` and `video_end_sec`

Each item in `gt_sampled_frame_boxes` contains a one-based `time_index` and a
0-1000 `bbox`. Charades-STA and ActivityNet JSONL annotations contain
`video_path`, `caption`, and `timestamp`. They can also include `frame_count`
and `fps`; otherwise the task reads this metadata from the video.

Run the spatio-temporal tasks with the PTD tube format:

```bash
export MODEL="/path/to/model"

cd "$LMMS_EVAL_REPO"
python -m accelerate.commands.launch -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args pretrained="$MODEL",ptd_root="$PTD_REPO",generation_format=spatio_temporal_grounding,system_prompt=none \
  --tasks ptd_vidstg,ptd_hcstvg \
  --batch_size 1 \
  --log_samples \
  --output_path ./ptd_stvg_results
```

Run the temporal-localization tasks with the temporal segment format:

```bash
python -m accelerate.commands.launch -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args pretrained="$MODEL",ptd_root="$PTD_REPO",generation_format=temporal_localization,system_prompt=none \
  --tasks ptd_charades_sta,ptd_activitynet \
  --batch_size 1 \
  --log_samples \
  --output_path ./ptd_temporal_results
```

lmms-eval computes and aggregates the task metrics during these runs. VidSTG
and HC-STVG report m_tIoU, m_vIoU, vIoU@0.3, and vIoU@0.5. Charades-STA and
ActivityNet report R@0.3, R@0.5, R@0.7, and mIoU.
