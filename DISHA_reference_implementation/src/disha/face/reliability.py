import cv2
import numpy as np

def variance_of_laplacian(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def mean_luminance(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray))
