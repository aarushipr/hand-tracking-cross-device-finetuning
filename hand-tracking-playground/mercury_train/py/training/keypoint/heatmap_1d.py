# Authors: Moses and Ryan - nothing stolen from the internet
import cv2
import numpy as np


def heatmap_1d(size_px: int, center_px: float, stddev_px: float) -> np.ndarray:
    dist_from_center: np.ndarray = np.arange(0, size_px, dtype=np.float32)
    dist_from_center += 0.5 - center_px

    variance = stddev_px * stddev_px

    return np.exp(- (dist_from_center)**2 / (2 * variance))


def two_heatmaps_to_2d_old(
        hmap_x: np.ndarray,
        hmap_y: np.ndarray) -> np.ndarray:
    hmap_2d_x = np.repeat(np.expand_dims(
        np.clip(hmap_x, 0, 1), 1).T, len(hmap_y), 0)
    hmap_2d_y = np.repeat(np.expand_dims(
        np.clip(hmap_y, 0, 1), 1), len(hmap_x), 1)

    hmap = np.clip((hmap_2d_x * hmap_2d_y), 0, 1)
    return hmap


def two_heatmaps_to_2d(hmap_x: np.ndarray, hmap_y: np.ndarray) -> np.ndarray:
    hmap_2d_x = np.expand_dims(np.clip(hmap_x, 0, 1), 1).T
    hmap_2d_y = np.expand_dims(np.clip(hmap_y, 0, 1), 1)

    # Multiplying a row by column vector - get a rank 1 wxh matrix
    return hmap_2d_y * hmap_2d_x


def test():
    hmap_x = heatmap_1d(256, 128, 50)

    hmap_y = heatmap_1d(256, 100, 50)

    cv2.imshow("hi", two_heatmaps_to_2d(hmap_x, hmap_y))
    cv2.waitKey(0)


if __name__ == "__main__":
    test()
