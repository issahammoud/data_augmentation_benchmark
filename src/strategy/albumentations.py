from typing import List
import tensorflow as tf
import albumentations as A
from functools import partial
from src.utils.enums import AugName
from src.strategy.aug_strategy import AugStrategy
from src.registry.aug_registry import AlbumentationsRegistry


class AlbumentationCompose(AugStrategy):
    AlbumentationsRegistry.register(AugName.CROP_RESIZE)(A.AtLeastOneBBoxRandomCrop)
    AlbumentationsRegistry.register(AugName.RESIZE)(A.Resize)
    AlbumentationsRegistry.register(AugName.H_FLIP)(A.HorizontalFlip)
    AlbumentationsRegistry.register(AugName.V_FLIP)(A.VerticalFlip)
    AlbumentationsRegistry.register(AugName.BLUR)(A.GaussianBlur)
    AlbumentationsRegistry.register(AugName.BRIGHTNESS)(
        partial(A.RandomBrightnessContrast, contrast_limit=0)
    )
    AlbumentationsRegistry.register(AugName.CONTRAST)(
        partial(A.RandomBrightnessContrast, brightness_limit=0)
    )

    def __init__(self, configs: List[dict]):
        augmentations = AlbumentationsRegistry.create_all(configs)
        self.compose = A.Compose(augmentations)

    def apply(self, img, mask):

        img, mask = tf.py_function(
            self._apply_fn, inp=[img, mask], Tout=[tf.float32, tf.int32]
        )
        img.set_shape(tf.TensorShape([None, None, 3]))
        mask.set_shape(tf.TensorShape([None, None, 1]))
        return img, mask

    def _apply_fn(self, img: tf.Tensor, mask: tf.Tensor):
        output = self.compose(image=img.numpy(), mask=mask.numpy())
        return output["image"], output["mask"]
