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
import pytesseract

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Marker detection
MARKER_MIN_AREA = 30.0  # Smallest contour area (in pixels) considered a valid red marker.
RED_HSV_RANGE_1 = (np.array([0, 100, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
# HSV lower/upper bounds capturing the lower portion of the red spectrum.
RED_HSV_RANGE_2 = (np.array([160, 100, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
# HSV lower/upper bounds capturing the upper portion of the red spectrum.
MARKER_KERNEL_SIZE = (5, 5)  # Elliptical kernel size for cleaning the red marker mask.
MARKER_OPEN_ITERATIONS = 2  # Number of opening operations to remove small noise in marker mask.
MARKER_DILATE_ITERATIONS = 1  # Number of dilation steps to merge fragmented marker regions.

# Contrast enhancement
CLAHE_CLIP_LIMIT = 2.5  # Clip limit for CLAHE; higher values increase local contrast more strongly.
CLAHE_TILE_GRID_SIZE = (8, 8)  # Grid size for CLAHE; smaller grids adapt more locally to contrast changes.

# Cropping
DEFAULT_CROP_PADDING = 10  # Extra pixels included around detected markers when cropping the region of interest.

# Dye area segmentation
GAUSSIAN_BLUR_KERNEL = (5, 5)  # Kernel size for smoothing the saturation channel before thresholding.
DYE_MASK_KERNEL_SIZE = (5, 5)  # Elliptical kernel size for morphological cleanup of the dye mask.
DYE_MASK_OPEN_ITERATIONS = 1  # Number of opening steps to remove isolated pixels in the dye mask.
DYE_MASK_CLOSE_ITERATIONS = 2  # Number of closing steps to fill small gaps in the dye mask.

# Digits region extraction
DEFAULT_DIGITS_WIDTH_RATIO = 0.35  # Horizontal proportion of the image captured for the bottom-right digits crop.
DEFAULT_DIGITS_HEIGHT_RATIO = 0.30  # Vertical proportion of the image captured for the bottom-right digits crop.

# Digit preprocessing
DIGIT_CLAHE_CLIP_LIMIT = 2.5  # CLAHE clip limit for digit preprocessing to sharpen contrast.
DIGIT_CLAHE_TILE_GRID_SIZE = (8, 8)  # CLAHE tile grid size for digit preprocessing.
DIGIT_SCALE_FACTOR = 2.0  # Upscaling factor applied before thresholding to improve OCR accuracy.
DIGIT_THRESH_KERNEL_SIZE = (3, 3)  # Kernel size for morphological closing on the digit mask.
DIGIT_MEDIAN_BLUR_SIZE = 3  # Median blur aperture size for removing salt-and-pepper noise after thresholding.

# OCR
TESSERACT_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789."  # OCR engine configuration string.

# Visualization
OVERLAY_COLOR = (0, 0, 255)  # BGR color for overlaying the dye mask on the original image.
OVERLAY_ALPHA = 0.4  # Opacity for the dye mask overlay; higher values increase mask prominence.
MARKER_ANNOTATION_COLOR = (0, 255, 0)  # BGR color for drawing marker annotations on preview images.
MARKER_CROSS_SIZE = 20  # Size of the cross marker drawn at detected marker centers.
MARKER_CROSS_THICKNESS = 2  # Line thickness of the cross marker overlay.
MARKER_CIRCLE_RADIUS = 10  # Radius of the circle drawn around each detected marker.
MARKER_CIRCLE_THICKNESS = 2  # Line thickness of the marker circle overlay.


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


def detect_red_markers(image: np.ndarray, min_area: float = MARKER_MIN_AREA) -> List[MarkerDetection]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = RED_HSV_RANGE_1
    lower_red2, upper_red2 = RED_HSV_RANGE_2

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MARKER_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MARKER_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=MARKER_DILATE_ITERATIONS)

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


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Improve local contrast while keeping colors realistic."""

    if image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def crop_between_markers(
    image: np.ndarray,
    markers: Sequence[MarkerDetection],
    padding: int = DEFAULT_CROP_PADDING,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for marker in markers:
        cv2.drawContours(mask, [marker.contour], contourIdx=-1, color=255, thickness=-1)

    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        raise RuntimeError("Could not determine region enclosed by red markers.")

    x, y, w, h = cv2.boundingRect(nonzero)
    x_min = max(x - padding, 0)
    y_min = max(y - padding, 0)
    x_max = min(x + w + padding, image.shape[1])
    y_max = min(y + h + padding, image.shape[0])

    cropped = image[y_min:y_max, x_min:x_max].copy()
    if cropped.size == 0:
        raise RuntimeError("Crop produced an empty image. Check marker positions.")

    cropped_mask = mask[y_min:y_max, x_min:x_max]
    masked_cropped = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)
    enhanced = enhance_contrast(masked_cropped)
    final_crop = cv2.bitwise_and(enhanced, enhanced, mask=cropped_mask)
    return final_crop, (x_min, y_min, x_max, y_max)


def compute_dye_area(image: np.ndarray) -> DyeAreaResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, GAUSSIAN_BLUR_KERNEL, 0)
    _, sat_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DYE_MASK_KERNEL_SIZE)
    cleaned = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel, iterations=DYE_MASK_OPEN_ITERATIONS)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=DYE_MASK_CLOSE_ITERATIONS)

    pixel_area = int(cv2.countNonZero(cleaned))
    total_pixels = image.shape[0] * image.shape[1]
    area_ratio = pixel_area / total_pixels if total_pixels else 0.0

    return DyeAreaResult(mask=cleaned, pixel_area=pixel_area, area_ratio=area_ratio)


def prepare_digits_region(
    image: np.ndarray,
    width_ratio: float = DEFAULT_DIGITS_WIDTH_RATIO,
    height_ratio: float = DEFAULT_DIGITS_HEIGHT_RATIO,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    x_start = max(int(w * (1.0 - width_ratio)), 0)
    y_start = max(int(h * (1.0 - height_ratio)), 0)
    return image[y_start:, x_start:], (x_start, y_start)


def preprocess_digits_region(
    region: np.ndarray,
    scale_factor: float = DIGIT_SCALE_FACTOR,
) -> Tuple[np.ndarray, float, float]:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=DIGIT_CLAHE_CLIP_LIMIT, tileGridSize=DIGIT_CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(gray)

    scaled = cv2.resize(
        enhanced,
        dsize=None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DIGIT_THRESH_KERNEL_SIZE)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.medianBlur(cleaned, DIGIT_MEDIAN_BLUR_SIZE)

    scale_x = cleaned.shape[1] / max(region.shape[1], 1)
    scale_y = cleaned.shape[0] / max(region.shape[0], 1)
    return cleaned, scale_x, scale_y


def run_ocr_on_region(region: np.ndarray) -> Tuple[List[OcrDetection], np.ndarray]:
    processed, scale_x, scale_y = preprocess_digits_region(region)

    data = pytesseract.image_to_data(processed, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)

    detections: List[OcrDetection] = []
    for text, conf, x, y, w, h in zip(
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    ):
        if not text.strip():
            continue
        try:
            conf_value = float(conf)
        except ValueError:
            conf_value = -1.0
        if conf_value < 0:
            continue
        cleaned_text = text.strip()
        if any(ch not in "0123456789." for ch in cleaned_text):
            continue

        x_orig = int(round(x / scale_x))
        y_orig = int(round(y / scale_y))
        w_orig = int(round(w / scale_x))
        h_orig = int(round(h / scale_y))
        detections.append(
            OcrDetection(text=cleaned_text, confidence=conf_value, bbox=(x_orig, y_orig, w_orig, h_orig))
        )

    return detections, processed


def save_digits_to_csv(detections: Sequence[OcrDetection], csv_path: Path, origin: Tuple[int, int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "confidence", "x", "y", "width", "height"])
        for det in detections:
            x_origin, y_origin = origin
            x, y, w, h = det.bbox
            writer.writerow([det.text, f"{det.confidence:.2f}", x + x_origin, y + y_origin, w, h])


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = OVERLAY_COLOR,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    mask_bool = mask.astype(bool)
    overlay = image.copy()
    overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)
    return overlay


def annotate_markers(image: np.ndarray, markers: Sequence[MarkerDetection]) -> np.ndarray:
    annotated = image.copy()
    for marker in markers:
        cv2.drawMarker(
            annotated,
            marker.center,
            color=MARKER_ANNOTATION_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=MARKER_CROSS_SIZE,
            thickness=MARKER_CROSS_THICKNESS,
        )
        cv2.circle(
            annotated,
            marker.center,
            radius=MARKER_CIRCLE_RADIUS,
            color=MARKER_ANNOTATION_COLOR,
            thickness=MARKER_CIRCLE_THICKNESS,
        )
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dye flow image with OpenCV.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("screenshot.png"), help="Path to the input image.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory where outputs will be saved.")
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_CROP_PADDING,
        help="Padding in pixels applied around detected markers when cropping.",
    )
    parser.add_argument(
        "--digits-width",
        type=float,
        default=DEFAULT_DIGITS_WIDTH_RATIO,
        dest="digits_width",
        help="Width ratio for bottom-right digits region crop.",
    )
    parser.add_argument(
        "--digits-height",
        type=float,
        default=DEFAULT_DIGITS_HEIGHT_RATIO,
        dest="digits_height",
        help="Height ratio for bottom-right digits region crop.",
    )
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

    digit_detections, processed_digits = run_ocr_on_region(digits_region)
    processed_digits_path = output_dir / "digits_region_processed.png"
    cv2.imwrite(str(processed_digits_path), processed_digits)
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
    print(f"Processed digits preview saved to: {processed_digits_path}")
    print(f"Numeric readings CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
