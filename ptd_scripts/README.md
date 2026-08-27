# PTD training

Prepare the annotations with the scripts described in
[`data/README.md`](../data/README.md), then fill in the exported paths at the
top of each script. Run the commands below from the repository root.

Run SFT directly from the Qwen3-VL model:

```bash
bash ptd_scripts/train_sft.sh
```

Merge the resulting adapter by setting `BASE_MODEL` to the Qwen3-VL model:

```bash
bash ptd_scripts/merge_lora.sh
```

Set `SFT_MODEL` in `train_grpo.sh` to that merged checkpoint and run GRPO:

```bash
bash ptd_scripts/train_grpo.sh
```

The GRPO adapter can be merged with the same merge script by setting
`BASE_MODEL` to the merged SFT checkpoint.
