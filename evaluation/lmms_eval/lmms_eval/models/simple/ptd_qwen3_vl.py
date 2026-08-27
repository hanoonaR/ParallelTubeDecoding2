"""Qwen3-VL lmms-eval adapter for PTD generation."""

from pathlib import Path
import sys

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("ptd_qwen3_vl")
class PTDQwen3VL(Qwen3_VL):
    def __init__(
        self,
        pretrained,
        ptd_root,
        generation_format="spatio_temporal_grounding",
        batch_size=1,
        **kwargs,
    ):
        if int(batch_size) != 1:
            raise ValueError("PTD evaluation uses batch_size=1.")
        if generation_format not in {
            "spatio_temporal_grounding",
            "temporal_localization",
        }:
            raise ValueError(f"Unsupported PTD generation format: {generation_format}")

        source = Path(ptd_root).expanduser().resolve() / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"PTD source directory not found: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        from train.monkey_patch_forward import replace_qwen3_with_ptd_forward

        replace_qwen3_with_ptd_forward()
        super().__init__(
            pretrained=pretrained,
            batch_size=1,
            temporal_patch_size=1,
            fps=2,
            max_num_frames=64,
            attn_implementation="sdpa",
            skip_special_tokens=False,
            use_video_time_tokens=False,
            **kwargs,
        )

        from dataset.data_utils import (
            patch_processor_with_time_tokens,
            patch_qwen3_video_processor,
        )

        patch_qwen3_video_processor(self.processor)
        patch_processor_with_time_tokens(self.processor)
        self.ptd_generation_format = generation_format

    def generate_until(self, requests):
        from model.ptd_generation import generate_ptd

        def collate(request):
            return -len(self.tokenizer.encode(request[0])), request[0]

        responses = []
        progress = tqdm(
            total=len(requests),
            disable=self.rank != 0,
            desc="Model Responding",
        )
        ordered = utils.Collator(
            [request.args for request in requests], collate, grouping=True
        )
        chunks = ordered.get_batched(n=1, batch_fn=None)

        for chunk in chunks:
            inputs, contexts, generation, until, _ = self._preprocess_chunk(chunk)
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
            generation = self._build_generate_kwargs(generation)
            if generation.get("num_beams", 1) != 1:
                raise ValueError("PTD does not use beam search.")

            completion, _ = generate_ptd(
                self.model,
                self.tokenizer,
                dict(inputs),
                max_new_tokens=generation["max_new_tokens"],
                max_time_tokens=int(inputs["video_grid_thw"][0, 0].item()),
                temperature=float(generation.get("temperature") or 0.0),
                top_p=generation.get("top_p"),
                top_k=generation.get("top_k"),
                generation_format=self.ptd_generation_format,
            )
            answers = self.processor.batch_decode(
                completion,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for answer, context in zip(answers, contexts):
                for stop in until:
                    if stop:
                        answer = answer.split(stop)[0]
                answer = self._strip_thinking(answer)
                responses.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, generation), answer
                )
                progress.update(1)

        progress.close()
        return ordered.get_original(responses)
