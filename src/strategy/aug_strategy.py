import tensorflow as tf
from abc import ABC, abstractmethod


class AugStrategy(ABC):
    @abstractmethod
    def apply(self, img: tf.Tensor, mask: tf.Tensor):
        raise NotImplementedError
