from dataclasses import dataclass
import numpy as np

'''
    Everything is in PIXEL COORDINATES.
'''
@dataclass
class bbox:
    cx: float = 0
    cy: float = 0
    w: float = 0
    h: float = 0

@dataclass
class ImageWithBoundingBoxes:
    image: np.ndarray
    bboxes: list # Better be two - left, right. None, or bbox.
