import os
import re
import glob
import tensorflow as tf
from abc import ABC, abstractmethod
from src.strategy.aug_strategy import AugStrategy


class AbstractLoader(ABC):
    def __init__(
        self,
        dataset: tf.data.Dataset,
        data_len: int,
        mode: str,
        batch_size: str,
        aug_compose: AugStrategy,
    ):
        self.dataset = dataset
        self._mode = mode
        self._length = data_len
        self._batch_size = batch_size
        self.aug_compose = aug_compose
        self.build_pipeline()

    def __len__(self):
        return self._length

    @classmethod
    @abstractmethod
    def from_source(cls, **kwargs):
        raise NotImplementedError

    @property
    def iteration_nb(self):
        return self._length // self._batch_size

    @staticmethod
    def get_data_path(imgs_path: str, gts_path: str, mode: str):
        all_images_paths = glob.glob(os.path.join(imgs_path, mode, "**/*"))

        all_gt_paths = []
        for img_path in all_images_paths:
            directory, name = img_path.split("/")[-2:]
            gt_name = re.sub("leftImg8bit", "gtFine_labelIds", name)
            gt_path = os.path.join(gts_path, mode, directory, gt_name)
            all_gt_paths.append(gt_path)

        return all_images_paths, all_gt_paths

    def _process_data(self, img: tf.Tensor, mask: tf.Tensor):
        img = tf.cast(img, tf.float32) / 255
        mask = tf.cast(mask, tf.int32)

        img, mask = self.aug_compose.apply(img, mask)
        return img, mask

    @abstractmethod
    def build_pipeline(self):
        raise NotImplementedError
