from enum import Enum, auto


class AugName(str, Enum):
    H_FLIP = "horizontal_flip"
    V_FLIP = "vertical_flip"
    RESIZE = "resize"
    CROP_RESIZE = "crop_and_resize"
    CONTRAST = "contrast"
    BRIGHTNESS = "brightness"
    BLUR = "blur"


class LoaderName(Enum):
    FROM_DATASET = auto()
    FROM_GENERATOR = auto()
    FROM_PYFUNC = auto()
