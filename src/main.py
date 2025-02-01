import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import warnings
import numpy as np
import tensorflow as tf

from src.registry.data_registry import DataRegistry
from src.strategy.tf_augmentations import TFCompose
from src.strategy.albumentations import AlbumentationCompose
from src.utils.utils import get_config, visualize_data


warnings.filterwarnings("ignore")


def consume_dataset(dataset: tf.data.Dataset):
    for _ in dataset:
        pass


def benchmark(loaders_list: list):
    for loaders in loaders_list:
        for loader in loaders:
            avr = []
            for _ in range(5):
                start_time = time.time()
                consume_dataset(loader.dataset)
                end_time = time.time()
                avr.append(end_time - start_time)
            print(
                f"loader {loader} generates {int(len(loader) / np.mean(avr))} image/s"
            )


if __name__ == "__main__":
    config = get_config()

    tf_compose = TFCompose(config["augmentation"])
    album_compose = AlbumentationCompose(config["augmentation"])

    loaders_with_tf = DataRegistry.create_all(
        config["img_dir"],
        config["gt_dir"],
        mode=config["mode"],
        batch_size=config["batch_size"],
        aug_compose=tf_compose,
    )
    loaders_with_album = DataRegistry.create_all(
        config["img_dir"],
        config["gt_dir"],
        mode=config["mode"],
        batch_size=config["batch_size"],
        aug_compose=album_compose,
    )
    if config["benchmark"]:
        benchmark([loaders_with_tf, loaders_with_album])

    if config["visualize"]:
        visualize_data(loaders_with_tf[0].dataset, loaders_with_album[0].dataset)
