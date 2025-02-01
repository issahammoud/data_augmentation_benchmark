import cv2
import yaml
import argparse
import numpy as np
import tensorflow as tf


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    return config


def visualize_data(dataset_1: tf.data.Dataset, dataset_2: tf.data.Dataset):
    stop = False
    for (batch_img_1, batch_mask_1), (batch_img_2, batch_mask_2) in zip(
        dataset_1, dataset_2
    ):
        for img_1, mask_1, img_2, mask_2 in zip(
            batch_img_1.numpy(),
            batch_mask_1.numpy(),
            batch_img_2.numpy(),
            batch_mask_2.numpy(),
        ):
            concat_imgs = np.concatenate([img_1 * 255, img_2 * 255], axis=0).astype(
                np.uint8
            )[..., ::-1]
            concat_masks = np.concatenate([mask_1, mask_2], axis=0).astype(np.uint8)
            concat_masks = apply_colormap(concat_masks)
            concat = np.concatenate([concat_imgs, concat_masks], axis=1)
            cv2.imshow("Augmentated Data", concat)
            key = cv2.waitKey(0)
            if key == 27:  # escape key
                stop = True
                break
        if stop:
            break
    cv2.destroyAllWindows()


def apply_colormap(mask: np.array, colormap: int = cv2.COLORMAP_JET):
    unique_classes = np.unique(mask)
    num_classes = len(unique_classes)

    if num_classes == 0:
        return np.zeros((*mask.shape[:2], 3), dtype=np.uint8)

    if num_classes == 1:
        return cv2.applyColorMap(np.zeros_like(mask), colormap)

    indices = np.searchsorted(unique_classes, mask)
    scaled = (indices * (255 / (num_classes - 1))).astype(np.uint8)
    return cv2.applyColorMap(scaled, colormap)
