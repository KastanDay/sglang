# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalProcessorOutput,
)
from sglang.srt.models.gemma4_audio import _SSCP_CONV_STRIDE_SIZES
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.models.gemma4_vision import get_gemma4_vision_pooling_indices
from sglang.srt.multimodal.processors.base_processor import (
    InvalidMultimodalDataError,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


def _get_visual_cache_modifier(modality, features, position_tensors):
    return {
        "position_tensors": position_tensors,
        "metadata": (
            modality,
            tuple((tuple(tensor.shape), str(tensor.dtype)) for tensor in features),
            tuple(
                (tuple(tensor.shape), str(tensor.dtype))
                for tensor in position_tensors
            ),
        ),
    }


class Gemma4SGLangProcessor(SGLangBaseProcessor):
    """Multimodal processor for Gemma4 supporting image, video, and audio inputs."""

    models = [Gemma4ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
            ),
            video_token="<|video|>",
            video_token_id=hf_config.video_token_id,
            video_token_regex=re.compile(
                r"<\|image>(?:<\|video\|>)+<image\|>|<\|video\|>"
            ),
            audio_token="<|audio|>",
            audio_token_id=hf_config.audio_token_id,
            audio_token_regex=re.compile(
                r"<\|audio>(?:<\|audio\|>)+<audio\|>|<\|audio\|>"
            ),
        ).build(_processor)

        # Register image-processor and video-processor outputs so they are stored on
        # MultimodalDataItem via collect_mm_items_from_processor_output.
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["video_position_ids"] = Modality.VIDEO

    def _get_audio_pad_multiple(self) -> int:
        """Derive the waveform padding alignment from processor config.

        The HF processor's ceil(duration_ms / audio_ms_per_token) formula can
        overshoot by 1 token relative to what the SSCP convolutions produce.
        Padding waveforms to a multiple of (hop_length * first_conv_stride)
        aligns the two calculations.
        See: gemma-4-eap-extras/examples/gemma-4-audio-examples.ipynb
        """
        fe = getattr(self._processor, "feature_extractor", None)
        hop = getattr(fe, "hop_length", 160)
        first_stride = _SSCP_CONV_STRIDE_SIZES[0][0]
        return hop * first_stride

    def _validate_visual_token_counts(self, mm_items) -> None:
        for item_idx, item in enumerate(mm_items):
            if item.is_image():
                modality = "image"
                position_attr = "image_position_ids"
            elif item.is_video():
                modality = "video"
                position_attr = "video_position_ids"
            else:
                continue
            if item.is_precomputed_embedding():
                continue

            position_values = getattr(item, position_attr, None)
            if item.feature is None:
                raise InvalidMultimodalDataError(
                    f"Gemma4 {modality} item {item_idx} has no processor feature"
                )
            try:
                features = [
                    torch.as_tensor(feature)
                    for feature in flatten_nested_list([item.feature])
                ]
                position_tensors = (
                    [
                        torch.as_tensor(position_ids)
                        for position_ids in flatten_nested_list([position_values])
                    ]
                    if position_values is not None
                    else []
                )
            except (RuntimeError, TypeError, ValueError) as e:
                raise InvalidMultimodalDataError(
                    f"Gemma4 {modality} item {item_idx} has invalid processor data: {e}"
                ) from e

            has_external_cache_key = (
                item.has_external_cache_key and item.hash is not None
            )
            item.has_external_cache_key = False
            if not has_external_cache_key:
                # Processor-output items are hashed before model validation.
                # Recompute internal keys with Gemma's position-aware modifier.
                item.hash = None
                item.pad_value = None
                item.set(
                    "_cache_key_modifier",
                    _get_visual_cache_modifier(modality, features, position_tensors),
                )
            if position_values is None:
                raise InvalidMultimodalDataError(
                    f"Gemma4 {modality} item {item_idx} is missing {position_attr}"
                )
            if len(features) != len(position_tensors):
                raise InvalidMultimodalDataError(
                    f"Gemma4 {modality} item {item_idx} has {len(features)} feature "
                    f"tensor(s) for {len(position_tensors)} position tensor(s)"
                )

            vision_config = getattr(self.hf_config, "vision_config", None)
            pooling_kernel_size = getattr(vision_config, "pooling_kernel_size", None)
            if not isinstance(pooling_kernel_size, int) or pooling_kernel_size <= 0:
                raise RuntimeError(
                    "Gemma4 vision config has invalid pooling_kernel_size="
                    f"{pooling_kernel_size!r}"
                )
            embedding_counts = []
            video_counts = []
            for feature, position_ids in zip(features, position_tensors, strict=True):
                if feature.ndim == 2:
                    feature = feature.unsqueeze(0)
                if position_ids.ndim == 2:
                    position_ids = position_ids.unsqueeze(0)
                if modality == "video" and feature.ndim == 4:
                    feature = feature.reshape(
                        -1, feature.shape[-2], feature.shape[-1]
                    )
                if modality == "video" and position_ids.ndim == 4:
                    position_ids = position_ids.reshape(
                        -1, position_ids.shape[-2], position_ids.shape[-1]
                    )
                if (
                    feature.ndim != 3
                    or position_ids.ndim != 3
                    or position_ids.shape[-1] != 2
                    or feature.shape[:2] != position_ids.shape[:2]
                ):
                    raise InvalidMultimodalDataError(
                        f"Gemma4 {modality} item {item_idx} has incompatible feature shape "
                        f"{tuple(feature.shape)} and {position_attr} shape "
                        f"{tuple(position_ids.shape)}"
                    )

                input_seq_len = feature.shape[1]
                output_length = input_seq_len // pooling_kernel_size**2
                if input_seq_len == output_length:
                    masks = (position_ids == -1).all(dim=-1)
                    tensor_counts = [mask.sum().item() for mask in masks]
                else:
                    try:
                        pooling_indices, _ = get_gemma4_vision_pooling_indices(
                            position_ids, input_seq_len, output_length
                        )
                    except ValueError as e:
                        raise InvalidMultimodalDataError(
                            f"Gemma4 {modality} item {item_idx} has invalid pooling "
                            f"geometry: {e}"
                        ) from e
                    if pooling_indices.max().item() >= output_length:
                        raise InvalidMultimodalDataError(
                            f"Gemma4 {modality} item {item_idx} has pooled position outside "
                            f"the valid range [0, {output_length})"
                        )
                    tensor_counts = [
                        torch.unique(indices).numel() for indices in pooling_indices
                    ]

                if modality == "video":
                    video_counts.append((tensor_counts, input_seq_len))
                else:
                    embedding_counts.extend(
                        (count, input_seq_len) for count in tensor_counts
                    )

            if modality == "video":
                if len(item.offsets or []) == len(video_counts):
                    embedding_counts = [
                        (sum(counts), len(counts) * input_seq_len)
                        for counts, input_seq_len in video_counts
                    ]
                else:
                    embedding_counts = [
                        (count, input_seq_len)
                        for counts, input_seq_len in video_counts
                        for count in counts
                    ]

            if item.offsets is None or len(item.offsets) != len(embedding_counts):
                raise InvalidMultimodalDataError(
                    f"Gemma4 {modality} item {item_idx} has {len(item.offsets or [])} "
                    f"token span(s) for {len(embedding_counts)} input(s)"
                )

            for input_idx, ((num_embeddings, num_patches), (start, end)) in enumerate(
                zip(embedding_counts, item.offsets, strict=True)
            ):
                num_placeholders = end - start + 1
                if num_placeholders != num_embeddings:
                    raise InvalidMultimodalDataError(
                        f"Gemma4 {modality} token count mismatch before scheduling: "
                        f"item={item_idx}, input={input_idx}, "
                        f"placeholders={num_placeholders}, embeddings={num_embeddings}, "
                        f"patches={num_patches}, "
                        f"pooling_kernel_size={pooling_kernel_size}"
                    )

    @staticmethod
    def _validate_precomputed_image_token_counts(mm_items) -> None:
        for item_idx, item in enumerate(mm_items):
            if not item.is_image() or not item.is_precomputed_embedding():
                continue

            if item.feature is None:
                raise InvalidMultimodalDataError(
                    f"Gemma4 precomputed image item {item_idx} has no embedding"
                )
            embedding = torch.as_tensor(item.feature)
            if embedding.ndim < 2:
                raise InvalidMultimodalDataError(
                    f"Gemma4 precomputed image item {item_idx} has invalid embedding "
                    f"shape {tuple(embedding.shape)}"
                )
            num_embeddings = embedding.reshape(-1, embedding.shape[-1]).shape[0]
            num_placeholders = sum(
                end - start + 1 for start, end in item.offsets or []
            )
            if num_placeholders != num_embeddings:
                raise InvalidMultimodalDataError(
                    "Gemma4 precomputed image token count mismatch: "
                    f"item={item_idx}, placeholders={num_placeholders}, "
                    f"embeddings={num_embeddings}"
                )

    def _video_decoder_to_tensor(self, vdw: VideoDecoderWrapper) -> torch.Tensor:
        """Convert a VideoDecoderWrapper to a (sampled_frames, C, H, W) uint8 tensor.

        SGLang's load_video returns VideoDecoderWrapper which the HF
        Gemma4VideoProcessor does not recognise (expects torch.Tensor or
        np.ndarray).  We replicate HF's uniform frame sampling here to
        avoid materialising the entire video in memory, then delegate the
        rest (resize, patchify, position IDs) to the HF video processor.
        """
        total = len(vdw)
        num_frames = getattr(
            getattr(self._processor, "video_processor", None),
            "num_frames",
            32,
        )
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = torch.arange(0, total, total / num_frames).int().tolist()
        frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if audios:
            pad_multiple = self._get_audio_pad_multiple()
            padded = []
            for a in audios:
                a = np.asarray(a)
                remainder = len(a) % pad_multiple
                if remainder != 0:
                    a = np.pad(a, (0, pad_multiple - remainder), mode="constant")
                padded.append(a)
            audios = padded
        if videos:
            videos = [
                (
                    self._video_decoder_to_tensor(v)
                    if isinstance(v, VideoDecoderWrapper)
                    else v
                )
                for v in videos
            ]
            kwargs.setdefault("do_sample_frames", False)
        return super().process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        """Process multimodal data including images, video, and audio."""
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data if request_obj else None,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        self._validate_visual_token_counts(mm_items)
        self._validate_precomputed_image_token_counts(mm_items)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
