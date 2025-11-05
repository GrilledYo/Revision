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
from typing import Iterable, List, Optional, Sequence, Tuple, Union

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
DEFAULT_CROP_PADDING =-20  # Extra pixels included around detected markers when cropping the region of interest.

# Dye area segmentation
GAUSSIAN_BLUR_KERNEL = (5, 5)  # Kernel size for smoothing the saturation channel before thresholding.
DYE_MASK_KERNEL_SIZE = (5, 5)  # Elliptical kernel size for morphological cleanup of the dye mask.
DYE_MASK_OPEN_ITERATIONS = 1  # Number of opening steps to remove isolated pixels in the dye mask.
DYE_MASK_CLOSE_ITERATIONS = 2  # Number of closing steps to fill small gaps in the dye mask.

# Digits region extraction
DEFAULT_DIGITS_WIDTH_RATIO = 0.25  # Horizontal proportion of the image captured for the bottom-right digits crop.
DEFAULT_DIGITS_HEIGHT_RATIO = 0.15  # Vertical proportion of the image captured for the bottom-right digits crop.

# Digit preprocessing
DIGIT_CLAHE_CLIP_LIMIT = 1 # CLAHE clip limit for digit preprocessing to sharpen contrast.
DIGIT_CLAHE_TILE_GRID_SIZE = (2, 2)  # CLAHE tile grid size for digit preprocessing.
DIGIT_SCALE_FACTOR = 2  # Upscaling factor applied before thresholding to improve OCR accuracy.
DIGIT_THRESH_KERNEL_SIZE = (1, 1)  # Kernel size for morphological closing on the digit mask.
DIGIT_MEDIAN_BLUR_SIZE = 1  # Median blur aperture size for removing salt-and-pepper noise after thresholding.
DIGIT_BORDER_PADDING = 10  # Extra white border (in pixels) added around the processed digit crop before OCR.

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


@dataclass
class SparseFlowResult:
    """Sparse optical flow vectors between two crops."""

    vectors: np.ndarray  # shape (N, 4) with columns [x, y, delta_x, delta_y]
    start_points: np.ndarray  # shape (N, 2) original feature coordinates
    end_points: np.ndarray  # shape (N, 2) tracked feature coordinates


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
    source: Optional[np.ndarray] = None,
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

    if source is None:
        source = image

    if source.shape[0] < y_max or source.shape[1] < x_max:
        raise ValueError("Source image is smaller than the computed crop bounds.")

    cropped = source[y_min:y_max, x_min:x_max].copy()
    if cropped.size == 0:
        raise RuntimeError("Crop produced an empty image. Check marker positions.")

    return cropped, (x_min, y_min, x_max, y_max)


def crop_with_bounds(image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the image using precomputed bounds."""

    x_min, y_min, x_max, y_max = bounds
    h, w = image.shape[:2]
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)
    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Invalid crop bounds for the given frame.")
    cropped = image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        raise RuntimeError("Cropping with the provided bounds produced an empty image.")
    return cropped.copy()


def compute_sparse_vector_field(reference: np.ndarray, target: np.ndarray) -> SparseFlowResult:
    """Compute sparse optical flow vectors between two cropped regions."""

    if reference.shape[:2] != target.shape[:2]:
        raise ValueError("Cropped regions must have matching spatial dimensions.")

    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prev_gray = to_grayscale(reference)
    next_gray = to_grayscale(target)

    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
    features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    if features is None:
        empty = np.empty((0, 4), dtype=np.float32)
        empty_points = np.empty((0, 2), dtype=np.float32)
        return SparseFlowResult(vectors=empty, start_points=empty_points, end_points=empty_points)

    flow, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if flow is None or status is None:
        empty = np.empty((0, 4), dtype=np.float32)
        empty_points = np.empty((0, 2), dtype=np.float32)
        return SparseFlowResult(vectors=empty, start_points=empty_points, end_points=empty_points)

    valid = status.flatten() == 1
    start_points = features[valid].reshape(-1, 2)
    end_points = flow[valid].reshape(-1, 2)

    if start_points.size == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        empty_points = np.empty((0, 2), dtype=np.float32)
        return SparseFlowResult(vectors=empty, start_points=empty_points, end_points=empty_points)

    displacements = end_points - start_points
    vectors = np.hstack((start_points, displacements)).astype(np.float32)
    return SparseFlowResult(
        vectors=vectors,
        start_points=start_points.astype(np.float32),
        end_points=end_points.astype(np.float32),
    )


def draw_sparse_flow(image: np.ndarray, flow: SparseFlowResult) -> np.ndarray:
    """Overlay sparse flow vectors on the reference crop."""

    overlay = image.copy()
    for start, disp in zip(flow.start_points, flow.vectors[:, 2:4]):
        x, y = start
        dx, dy = disp
        pt1 = (int(round(x)), int(round(y)))
        pt2 = (int(round(x + dx)), int(round(y + dy)))
        cv2.arrowedLine(overlay, pt1, pt2, (0, 255, 0), thickness=1, tipLength=0.3)
    return overlay


def save_sparse_flow_sequence_to_csv(flow_vectors: Iterable[np.ndarray], csv_path: Path) -> None:
    """Persist a sequence of sparse flow vectors to CSV with boundary markers."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "delta_x", "delta_y", "new_field"])
        for vectors in flow_vectors:
            if vectors.size == 0:
                continue
            for idx, (x, y, dx, dy) in enumerate(vectors):
                marker = 1 if idx == 0 else 0
                writer.writerow(
                    [f"{x:.6f}", f"{y:.6f}", f"{dx:.6f}", f"{dy:.6f}", marker]
                )


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
) -> Tuple[np.ndarray, float, float, int]:
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

    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DIGIT_THRESH_KERNEL_SIZE)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.medianBlur(cleaned, DIGIT_MEDIAN_BLUR_SIZE)

    scale_x = cleaned.shape[1] / max(region.shape[1], 1)
    scale_y = cleaned.shape[0] / max(region.shape[0], 1)

    padded = cv2.copyMakeBorder(
        cleaned,
        DIGIT_BORDER_PADDING,
        DIGIT_BORDER_PADDING,
        DIGIT_BORDER_PADDING,
        DIGIT_BORDER_PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    return padded, scale_x, scale_y, DIGIT_BORDER_PADDING


def run_ocr_on_region(region: np.ndarray) -> Tuple[List[OcrDetection], np.ndarray]:
    processed, scale_x, scale_y, padding = preprocess_digits_region(region)

    def collect_detections(image: np.ndarray) -> List[OcrDetection]:
        data = pytesseract.image_to_data(image, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
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

            x_unpadded = max(x - padding, 0)
            y_unpadded = max(y - padding, 0)
            x_orig = int(round(x_unpadded / scale_x))
            y_orig = int(round(y_unpadded / scale_y))
            w_orig = int(round(w / scale_x))
            h_orig = int(round(h / scale_y))
            detections.append(
                OcrDetection(text=cleaned_text, confidence=conf_value, bbox=(x_orig, y_orig, w_orig, h_orig))
            )
        return detections

    for candidate in (processed, cv2.bitwise_not(processed)):
        detections = collect_detections(candidate)
        if detections:
            return detections, candidate

    return [], processed


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
    parser = argparse.ArgumentParser(description="Analyze dye flow feed with OpenCV.")
    parser.add_argument(
        "--image",
        type=Path,
        help=(
            "Path to a still image to analyze. When provided, the script will skip video capture "
            "and process the supplied file instead."
        ),
    )
    parser.add_argument(
        "--camera",
        default="0",
        help=(
            "Virtual camera identifier to read frames from. Provide an integer index or a "
            "device path such as /dev/video2."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Directory where outputs will be saved."
    )
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

    image: np.ndarray
    cap: Optional[cv2.VideoCapture] = None

    if args.image is not None:
        image = load_image(args.image)
    else:
        camera_source: Union[int, str]
        if isinstance(args.camera, str) and args.camera.isdigit():
            camera_source = int(args.camera)
        else:
            camera_source = args.camera

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError(
                "Unable to open camera source {src}. If you intended to analyze a still image, pass it via --image.".format(src=args.camera)
            )

        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            raise RuntimeError("Video file does not contain any frames.")

        image = frame
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    dye_area = compute_dye_area(image)
    dye_mask_path = output_dir / "dye_mask.png"
    cv2.imwrite(str(dye_mask_path), dye_area.mask)
    overlay_path = output_dir / "dye_overlay.png"
    overlay = overlay_mask(image, dye_area.mask)
    cv2.imwrite(str(overlay_path), overlay)

    markers = detect_red_markers(image)
    cropped_color, crop_bounds = crop_between_markers(image, markers, padding=args.padding)
    cropped_mask, _ = crop_between_markers(image, markers, padding=args.padding, source=dye_area.mask)
    previous_crop = cropped_color

    cropped_path = output_dir / "cropped_region.png"
    cv2.imwrite(str(cropped_path), cropped_color)
    cropped_mask_path = output_dir / "cropped_region_mask.png"
    cv2.imwrite(str(cropped_mask_path), cropped_mask)

    flow_vectors_sequence: List[np.ndarray] = []
    total_frames = 1
    frame_pairs = 0
    total_vectors = 0
    flow_vis_path = output_dir / "sparse_flow_visualization.png"
    flow_overlay_saved = False

    if cap is not None:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break
            total_frames += 1
            current_crop = crop_with_bounds(frame, crop_bounds)
            flow_result = compute_sparse_vector_field(previous_crop, current_crop)
            flow_vectors_sequence.append(flow_result.vectors)
            frame_pairs += 1
            total_vectors += flow_result.vectors.shape[0]
            if flow_result.vectors.size and not flow_overlay_saved:
                flow_overlay = draw_sparse_flow(previous_crop, flow_result)
                cv2.imwrite(str(flow_vis_path), flow_overlay)
                flow_overlay_saved = True
            previous_crop = current_crop

        cap.release()

    flow_csv_path = output_dir / "sparse_flow_vectors.csv"
    save_sparse_flow_sequence_to_csv(flow_vectors_sequence, flow_csv_path)

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
        summary_file.write("\nFeed analysis:\n")
        if args.image is not None:
            summary_file.write(f"Image source: {args.image}\n")
        else:
            summary_file.write(f"Camera source: {args.camera}\n")
        summary_file.write(f"Total frames processed: {total_frames}\n")
        summary_file.write(f"Frame pairs analyzed: {frame_pairs}\n")
        summary_file.write(f"Total vectors exported: {total_vectors}\n")
        summary_file.write(
            "Vectors stored per frame pair with columns [x, y, delta_x, delta_y, new_field].\n"
        )
        summary_file.write(f"CSV export: {flow_csv_path.name}\n")
        if flow_overlay_saved:
            summary_file.write(f"First visualization: {flow_vis_path.name}\n")
        else:
            summary_file.write("No sparse flow visualization generated (no vectors detected).\n")
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
    print(f"Camera source analyzed: {args.camera}")
    print(f"Cropped region saved to: {cropped_path}")
    print(f"Cropped mask saved to: {cropped_mask_path}")
    print(f"Dye mask saved to: {dye_mask_path}")
    print(f"Dye overlay saved to: {overlay_path}")
    print(f"Sparse flow CSV saved to: {flow_csv_path}")
    if flow_overlay_saved:
        print(f"Sparse flow visualization saved to: {flow_vis_path}")
    else:
        print("No sparse flow visualization generated (no vectors detected).")
    print(f"Digits region saved to: {digits_region_path}")
    print(f"Processed digits preview saved to: {processed_digits_path}")
    print(f"Numeric readings CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
