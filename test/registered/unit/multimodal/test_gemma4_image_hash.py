"""Regression tests for Gemma 4 image cache identity."""

import unittest

import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items, hash_feature
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    _compute_pad_value,
)
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGemma4ImageHash(CustomTestCase):
    def test_position_ids_are_part_of_image_hash(self):
        processor = Gemma4SGLangProcessor.__new__(Gemma4SGLangProcessor)
        processor.ATTR_NAME_TO_MODALITY = {
            "pixel_values": Modality.IMAGE,
            "image_position_ids": Modality.IMAGE,
        }
        processor.FEATURE_NAMES = ["pixel_values"]
        mm_items = processor.collect_mm_items_from_processor_output(
            {
                "pixel_values": torch.zeros((2, 4, 3)),
                "image_position_ids": torch.tensor(
                    [
                        [[0, 0], [1, 0], [0, 1], [1, 1]],
                        [[0, 0], [2, 0], [0, 1], [2, 1]],
                    ]
                ),
            }
        )
        mm_items[0].offsets = [(0, 1), (2, 3)]
        first, second = get_new_expanded_mm_items(mm_items)
        self.assertIsNone(first.hash)
        self.assertIsNone(second.hash)

        first.set_pad_value()
        second.set_pad_value()

        self.assertNotEqual(first.hash, second.hash)

        standalone = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=first.feature.clone(),
            model_specific_data={
                "image_position_ids": first.image_position_ids.clone()
            },
        )
        Gemma4SGLangProcessor._set_position_aware_hash_fields([standalone])
        standalone.set_pad_value()
        self.assertEqual(first.hash, standalone.hash)

    def test_audio_mask_is_part_of_hash(self):
        feature = torch.zeros((1, 2, 3))
        first = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=feature.clone(),
            model_specific_data={"input_features_mask": torch.tensor([[True, False]])},
        )
        second = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=feature.clone(),
            model_specific_data={"input_features_mask": torch.tensor([[True, True]])},
        )

        Gemma4SGLangProcessor._set_position_aware_hash_fields([first, second])
        first.set_pad_value()
        second.set_pad_value()

        self.assertNotEqual(first.hash, second.hash)

    def test_video_position_ids_are_part_of_hash(self):
        feature = torch.zeros((1, 2, 3))
        first = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=feature.clone(),
            model_specific_data={
                "video_position_ids": torch.tensor([[[0, 0], [1, 0]]])
            },
        )
        second = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=feature.clone(),
            model_specific_data={
                "video_position_ids": torch.tensor([[[0, 0], [2, 0]]])
            },
        )

        Gemma4SGLangProcessor._set_position_aware_hash_fields([first, second])
        first.set_pad_value()
        second.set_pad_value()

        self.assertNotEqual(first.hash, second.hash)

    def test_position_aware_hash_updates_pad_value(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros((1, 2, 3)),
            model_specific_data={
                "image_position_ids": torch.tensor([[[0, 0], [1, 0]]])
            },
        )

        Gemma4SGLangProcessor._set_position_aware_hash_fields([item])
        item.set_pad_value()

        self.assertEqual(item.pad_value, _compute_pad_value(item.hash))

    def test_default_hash_is_unchanged(self):
        feature = torch.arange(6).reshape(1, 2, 3)
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=feature)

        item.set_pad_value()

        self.assertEqual(item.hash, hash_feature(feature))

    def test_precomputed_embedding_does_not_hash_position_ids(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            precomputed_embeddings=torch.zeros((2, 3)),
            model_specific_data={
                "image_position_ids": torch.tensor([[[0, 0], [1, 0]]])
            },
        )

        Gemma4SGLangProcessor._set_position_aware_hash_fields([item])

        self.assertEqual(item.hash_feature_fields, ())


if __name__ == "__main__":
    unittest.main()
