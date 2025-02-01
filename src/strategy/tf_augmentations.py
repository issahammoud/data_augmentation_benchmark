from typing import List
import tensorflow as tf
from abc import ABC, abstractmethod

from src.utils.enums import AugName
from src.strategy.aug_strategy import AugStrategy
from src.registry.aug_registry import TFAugRegistry


class TFCompose(AugStrategy):
    def __init__(self, configs: List[dict]):
        tf.random.set_seed(10452)
        self.aug_list = TFAugRegistry.create_all(configs)

    def apply(self, img: tf.Tensor, mask: tf.Tensor):
        for augmentation in self.aug_list:
            img, mask = augmentation.apply(img, mask)

        return img, mask


class Augmentation(ABC):
    def __init__(self, p: float):
        self._apply_proba = p
        assert 0.0 <= p <= 1.0, f"apply_proba should be in [0-1], however, it is {p}"

    @abstractmethod
    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        raise NotImplementedError

    def apply(self, img: tf.Tensor, mask: tf.Tensor):
        proba = tf.random.uniform(shape=[], maxval=1.0)
        return tf.cond(
            proba < self._apply_proba,
            true_fn=lambda: self(img, mask),
            false_fn=lambda: (img, mask),
        )


@TFAugRegistry.register(AugName.CROP_RESIZE)
class CropResize(Augmentation):
    def __init__(self, height: int, width: int, p: float):
        super().__init__(p)
        self._height, self._width = height, width

    def __call__(self, img: tf.Tensor, mask: tf.Tensor):

        y1_x1 = tf.random.uniform((1, 2), minval=0, maxval=0.5)
        y2_x2 = tf.random.uniform((1, 2), minval=0.5, maxval=1)
        box = tf.concat([y1_x1, y2_x2], axis=1)
        box_id = tf.constant([0], tf.int32)
        img = tf.image.crop_and_resize(
            tf.expand_dims(img, axis=0), box, box_id, (self._height, self._width)
        )[0]
        mask = tf.image.crop_and_resize(
            tf.expand_dims(mask, axis=0),
            box,
            box_id,
            (self._height, self._width),
            tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )[0]
        mask = tf.cast(mask, tf.int32)
        return tf.clip_by_value(img, 0, 1), mask


@TFAugRegistry.register(AugName.RESIZE)
class Resize(Augmentation):
    def __init__(self, height: int, width: int, p: float):
        super().__init__(p)
        self._shape = height, width

    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        img = tf.image.resize(img, self._shape)
        mask = tf.image.resize(
            mask, self._shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return img, mask


@TFAugRegistry.register(AugName.H_FLIP)
class HFlip(Augmentation):
    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        return tf.image.flip_left_right(img), tf.image.flip_left_right(mask)


@TFAugRegistry.register(AugName.V_FLIP)
class VFlip(Augmentation):
    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        return tf.image.flip_up_down(img), tf.image.flip_up_down(mask)


@TFAugRegistry.register(AugName.CONTRAST)
class AdjustContrast(Augmentation):
    def __init__(self, contrast_limit: float, p: float):
        super().__init__(p)
        self._contrast_factor = contrast_limit

    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        adjusted_img = tf.image.adjust_contrast(
            img, contrast_factor=self._contrast_factor
        )
        return tf.clip_by_value(adjusted_img, 0, 1), mask


@TFAugRegistry.register(AugName.BRIGHTNESS)
class AdjustBrightness(Augmentation):
    def __init__(self, brightness_limit: float, p: float):
        super().__init__(p)
        self._delta = brightness_limit

    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        adjusted_img = tf.image.adjust_brightness(img, delta=self._delta)
        return tf.clip_by_value(adjusted_img, 0, 1), mask


@TFAugRegistry.register(AugName.BLUR)
class Blur(Augmentation):
    def __init__(self, blur_limit: int, sigma_limit: float, p: float):
        super().__init__(p)
        self._kernel_size = blur_limit
        self._sigma = sigma_limit

    def _compute_gauss_kernel(self, channels: int):
        """
        the function source:
        https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
        """
        kernel_size = self._kernel_size
        sigma = self._sigma
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    def __call__(self, img: tf.Tensor, mask: tf.Tensor):
        gaussian_kernel = self._compute_gauss_kernel(tf.shape(img)[-1])
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]
        img = img[tf.newaxis, ...]

        blurred_img = tf.nn.depthwise_conv2d(
            img, gaussian_kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
        )[0]

        return tf.clip_by_value(blurred_img, 0, 1), mask
