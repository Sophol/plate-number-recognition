import cv2
import numpy as np

# Canonical plate size after warping; wide enough for OCR on the middle zone.
PLATE_WIDTH = 320
PLATE_HEIGHT = 160


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(coordinate_sum)]  # smallest x+y
    ordered[2] = points[np.argmax(coordinate_sum)]  # largest x+y

    diff = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(diff)]  # smallest y-x
    ordered[3] = points[np.argmax(diff)]
    return ordered


def warp_plate(
    image: np.ndarray,
    corners: np.ndarray | None,
    width: int = PLATE_WIDTH,
    height: int = PLATE_HEIGHT,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Rectify a plate to a canonical rectangle.

    Falls back to a plain resize when corners are unavailable, so a box-only
    detector still produces a usable crop.

    Detectors report corners in frame coordinates, so callers that pass a
    cropped image must pass the crop's top-left as ``origin``. Without it the
    transform maps outside the crop and yields an all-black plate, which makes
    OCR silently return nothing.
    """
    if corners is None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    if origin != (0, 0):
        corners = corners - np.asarray(origin, dtype=np.float32)

    source = order_corners(corners)
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (width, height))


def split_zones(plate: np.ndarray) -> dict[str, np.ndarray]:
    """Split a rectified plate into its three horizontal bands.

    Cambodian layout: Khmer province on top, Latin/digits in the middle,
    English province at the bottom.
    """
    h = plate.shape[0]
    return {
        "top": plate[: int(h * 0.30)],
        "middle": plate[int(h * 0.25) : int(h * 0.78)],
        "bottom": plate[int(h * 0.75) :],
    }
