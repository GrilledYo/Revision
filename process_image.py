"""Image analysis pipeline for dye flow screenshot.

This script detects red corner markers, crops the region they delimit,
estimates the dye-covered area, and extracts numeric readouts from the
bottom-right of the image. Outputs are written into the specified output
directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# NumPy removed the deprecated ``np.bool`` alias in version 1.24. Some of the
# libraries used by this script (notably certain OpenCV builds) still reference
# the alias internally, which raises an ``AttributeError`` at runtime. To retain
# compatibility across NumPy versions, recreate the alias when it is missing by
# pointing it at the canonical ``np.bool_`` scalar type.
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]
import pytesseract


@dataclass
class MarkerDetection:
    """Data structure storing marker information."""

    center: Tuple[int, int]
    area: float
    contour: np.ndarray


@dataclass
class DyeAreaResult:
    """Summary of dye segmentation."""

    mask: np.ndarray
    pixel_area: int
    area_ratio: float


@dataclass
class OcrDetection:
    """OCR result for a single token."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return image


def detect_red_markers(image: np.ndarray, min_area: float = 30.0) -> List[MarkerDetection]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def make_bound(values: Sequence[int]) -> np.ndarray:
        """Create a contiguous HSV bound matching the source image dtype."""

        bound = np.asarray(values, dtype=hsv.dtype)
        # OpenCV expects the thresholds to be treated as scalars, but newer
        # versions are stricter about the underlying array type. Ensuring a
        # contiguous `ndarray` avoids the "lowerb is not a numpy array" error
        # triggered when the bounds come through as Python sequences under
        # certain NumPy/OpenCV combinations.
        return np.ascontiguousarray(bound.reshape(1, 1, -1))

    lower_red1 = make_bound((0, 100, 80))
    upper_red1 = make_bound((10, 255, 255))
    lower_red2 = make_bound((160, 100, 80))
    upper_red2 = make_bound((179, 255, 255))

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[MarkerDetection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        detections.append(MarkerDetection(center=(cx, cy), area=area, contour=contour))

    detections.sort(key=lambda d: d.area, reverse=True)
    if len(detections) < 2:
        raise RuntimeError("Expected at least two red markers but found fewer."
                           "Please verify the image or adjust detection parameters.")

    return detections[:2]


def crop_between_markers(image: np.ndarray, markers: Sequence[MarkerDetection], padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    xs = [m.center[0] for m in markers]
    ys = [m.center[1] for m in markers]
    x_min = max(min(xs) - padding, 0)
    y_min = max(min(ys) - padding, 0)
    x_max = min(max(xs) + padding, image.shape[1])
    y_max = min(max(ys) + padding, image.shape[0])

    cropped = image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        raise RuntimeError("Crop produced an empty image. Check marker positions.")
    return cropped, (x_min, y_min, x_max, y_max)


def compute_dye_area(image: np.ndarray) -> DyeAreaResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, sat_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    pixel_area = int(cv2.countNonZero(cleaned))
    total_pixels = image.shape[0] * image.shape[1]
    area_ratio = pixel_area / total_pixels if total_pixels else 0.0

    return DyeAreaResult(mask=cleaned, pixel_area=pixel_area, area_ratio=area_ratio)


def prepare_digits_region(image: np.ndarray, width_ratio: float = 0.35, height_ratio: float = 0.30) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    x_start = max(int(w * (1.0 - width_ratio)), 0)
    y_start = max(int(h * (1.0 - height_ratio)), 0)
    return image[y_start:, x_start:], (x_start, y_start)


def run_ocr_on_region(region: np.ndarray) -> List[OcrDetection]:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 11)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789."
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    detections: List[OcrDetection] = []
    for text, conf, x, y, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
        if not text.strip():
            continue
        try:
            conf_value = float(conf)
        except ValueError:
            conf_value = -1.0
        cleaned_text = text.strip()
        if any(ch not in "0123456789." for ch in cleaned_text):
            continue
        detections.append(OcrDetection(text=cleaned_text, confidence=conf_value, bbox=(x, y, w, h)))

    return detections


def save_digits_to_csv(detections: Sequence[OcrDetection], csv_path: Path, origin: Tuple[int, int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "confidence", "x", "y", "width", "height"])
        for det in detections:
            x_origin, y_origin = origin
            x, y, w, h = det.bbox
            writer.writerow([det.text, f"{det.confidence:.2f}", x + x_origin, y + y_origin, w, h])


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    mask_bool = mask.astype(bool)
    overlay = image.copy()
    overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)
    return overlay


def annotate_markers(image: np.ndarray, markers: Sequence[MarkerDetection]) -> np.ndarray:
    annotated = image.copy()
    for marker in markers:
        cv2.drawMarker(annotated, marker.center, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.circle(annotated, marker.center, radius=10, color=(0, 255, 0), thickness=2)
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dye flow image with OpenCV.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("screenshot.png"), help="Path to the input image.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory where outputs will be saved.")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels applied around detected markers when cropping.")
    parser.add_argument("--digits-width", type=float, default=0.35, dest="digits_width", help="Width ratio for bottom-right digits region crop.")
    parser.add_argument("--digits-height", type=float, default=0.30, dest="digits_height", help="Height ratio for bottom-right digits region crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = load_image(args.image)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    markers = detect_red_markers(image)
    cropped, crop_bounds = crop_between_markers(image, markers, padding=args.padding)
    cropped_path = output_dir / "cropped_region.png"
    cv2.imwrite(str(cropped_path), cropped)

    dye_area = compute_dye_area(image)
    dye_mask_path = output_dir / "dye_mask.png"
    cv2.imwrite(str(dye_mask_path), dye_area.mask)
    overlay_path = output_dir / "dye_overlay.png"
    overlay = overlay_mask(image, dye_area.mask)
    cv2.imwrite(str(overlay_path), overlay)

    annotated_markers = annotate_markers(image, markers)
    markers_path = output_dir / "marker_detection.png"
    cv2.imwrite(str(markers_path), annotated_markers)

    digits_region, origin = prepare_digits_region(image, width_ratio=args.digits_width, height_ratio=args.digits_height)
    digits_region_path = output_dir / "digits_region.png"
    cv2.imwrite(str(digits_region_path), digits_region)

    digit_detections = run_ocr_on_region(digits_region)
    csv_path = output_dir / "numeric_readings.csv"
    save_digits_to_csv(digit_detections, csv_path, origin)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Dye area analysis\n")
        summary_file.write(f"Total pixels: {image.shape[0] * image.shape[1]}\n")
        summary_file.write(f"Dye pixel count: {dye_area.pixel_area}\n")
        summary_file.write(f"Dye area ratio: {dye_area.area_ratio:.4f}\n")
        summary_file.write("\n")
        summary_file.write("Crop bounds (x_min, y_min, x_max, y_max):\n")
        summary_file.write(
            f"{crop_bounds[0]}, {crop_bounds[1]}, {crop_bounds[2]}, {crop_bounds[3]}\n"
        )
        summary_file.write("\nNumeric detections:\n")
        if digit_detections:
            for det in digit_detections:
                x, y, w, h = det.bbox
                summary_file.write(
                    f"text={det.text} confidence={det.confidence:.2f} bbox={(x + origin[0], y + origin[1], w, h)}\n"
                )
        else:
            summary_file.write("No numeric text detected.\n")

    print("Analysis complete.")
    print(f"Cropped region saved to: {cropped_path}")
    print(f"Dye mask saved to: {dye_mask_path}")
    print(f"Dye overlay saved to: {overlay_path}")
    print(f"Digits region saved to: {digits_region_path}")
    print(f"Numeric readings CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
