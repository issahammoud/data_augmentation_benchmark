import cv2
import numpy as np
import tensorflow as tf

from src.utils.enums import LoaderName
from src.registry.data_registry import DataRegistry
from src.dataloaders.abstract_loader import AbstractLoader


@DataRegistry.register(LoaderName.FROM_DATASET)
class CityscapesFromDataset(AbstractLoader):

    @classmethod
    def from_source(cls, imgs_path: str, gts_path: str, mode: str, **kwargs):
        data_path_list = AbstractLoader.get_data_path(imgs_path, gts_path, mode)
        dataset = tf.data.Dataset.from_tensor_slices(data_path_list)
        return cls(dataset, len(data_path_list[0]), mode, **kwargs)

    def _read_data(self, img_path: tf.Tensor, mask_path: tf.Tensor):
        img = tf.io.decode_png(tf.io.read_file(img_path), channels=3, dtype=tf.uint8)
        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1, dtype=tf.uint8)
        return img, mask

    def build_pipeline(self):
        self.dataset = self.dataset.map(
            self._read_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.dataset = self.dataset.map(
            self._process_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.dataset = self.dataset.batch(
            self._batch_size, num_parallel_calls=tf.data.AUTOTUNE
        )


@DataRegistry.register(LoaderName.FROM_PYFUNC)
class CityscapesFromPyFunction(AbstractLoader):

    @classmethod
    def from_source(cls, imgs_path: str, gts_path: str, mode: str, **kwargs):
        data_path_list = AbstractLoader.get_data_path(imgs_path, gts_path, mode)
        dataset = tf.data.Dataset.from_tensor_slices(data_path_list)
        return cls(dataset, len(data_path_list[0]), mode, **kwargs)

    def _read_data(self, img_path: tf.Tensor, mask_path: tf.Tensor):
        img = cv2.imread(img_path.numpy().decode("ascii"), -1)[..., ::-1]
        mask = cv2.imread(mask_path.numpy().decode("ascii"), -1)[..., np.newaxis]
        return img, mask

    def _py_function(self, img_path: tf.Tensor, mask_path: tf.Tensor):
        img, mask = tf.py_function(
            self._read_data, inp=[img_path, mask_path], Tout=[tf.uint8, tf.uint8]
        )
        img.set_shape(tf.TensorShape([None, None, 3]))
        mask.set_shape(tf.TensorShape([None, None, 1]))
        return img, mask

    def build_pipeline(self):
        self.dataset = self.dataset.map(
            self._py_function, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.dataset = self.dataset.map(
            self._process_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.dataset = self.dataset.batch(
            self._batch_size, num_parallel_calls=tf.data.AUTOTUNE
        )


@DataRegistry.register(LoaderName.FROM_GENERATOR)
class CityscapesFromGenerator(AbstractLoader):

    @classmethod
    def from_source(cls, imgs_path: str, gts_path: str, mode: str, **kwargs):
        data_path_list = AbstractLoader.get_data_path(imgs_path, gts_path, mode)

        def generator():
            for img_path, mask_path in zip(*data_path_list):
                img = cv2.imread(img_path, -1)[..., ::-1]
                mask = cv2.imread(mask_path, -1)[..., np.newaxis]

                yield img, mask

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.uint8),
            ),
        )
        return cls(dataset, len(data_path_list[0]), mode, **kwargs)

    def build_pipeline(self):
        self.dataset = self.dataset.map(
            self._process_data, num_parallel_calls=tf.data.AUTOTUNE
        )
        self.dataset = self.dataset.batch(
            self._batch_size, num_parallel_calls=tf.data.AUTOTUNE
        )
