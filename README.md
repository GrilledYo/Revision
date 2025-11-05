# Dye Flow Image Analysis

This repository contains a Python script that performs a computer-vision analysis of the provided dye-flow `screenshot.png`.

The script:

1. Detects the two red corner markers in the image and crops the rectangular region they delimit.
2. Segments the dyed flow to estimate its total area in pixels and as a percentage of the full frame.
3. Extracts the numeric readout located near the bottom-right of the image and saves the OCR results to a CSV file.
4. Writes several diagnostic images and a text summary to help verify the intermediate steps.

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system and available on the PATH. On Windows you can use the [prebuilt installer](https://github.com/UB-Mannheim/tesseract/wiki), while macOS users can install it via Homebrew (`brew install tesseract`).
- Python packages listed in `requirements.txt`.

Install the dependencies in your Spyder environment:

```bash
pip install -r requirements.txt
```

If Tesseract is installed in a non-standard location, update the script by setting `pytesseract.pytesseract.tesseract_cmd` to the full executable path near the top of the file.

## Usage

Run the analysis from the repository root. By default the script connects to camera index `0`:

```bash
python process_image.py
```

If you want to process a still image (for example the provided `screenshot.png`), pass the path with `--image` and the camera will
be skipped entirely:

```bash
python process_image.py --image screenshot.png
```

To use a specific virtual camera (for example `/dev/video2`), pass the `--camera` option:

```bash
python process_image.py --camera /dev/video2
```

Key command line options:

- `--image`: Path to a single image file. When supplied, the script does not attempt to open a camera.
- `--camera`: Virtual camera identifier. Accepts an integer index or device path.
- `--output`: Directory where results are stored (default: `outputs/`).
- `--padding`: Extra pixels to add around the detected markers when cropping.
- `--digits-width`, `--digits-height`: Ratios that control how much of the bottom-right corner is used for OCR. Increase them if the numbers are not fully captured.

All generated artifacts are written to the output directory:

- `cropped_region.png`: Color crop between the detected markers in the primary image.
- `marker_detection.png`: Visualization of the detected marker centers.
- `dye_mask.png`: Binary mask showing the detected dye.
- `dye_overlay.png`: Original image with the dye mask overlaid for quick inspection.
- `digits_region.png`: The bottom-right crop used for OCR.
- `numeric_readings.csv`: CSV containing the recognized numbers, OCR confidence, and bounding boxes in the original image coordinates.
- `summary.txt`: Text file summarizing the area statistics and OCR results.
- `cropped_region_mask.png`: Cropped dye mask corresponding to the marker region.
- `sparse_flow_vectors.csv`: Sparse optical flow vectors stored as float32 values in `[x, y, delta_x, delta_y]` format.
- `sparse_flow_visualization.png`: Reference crop overlaid with the sparse vector field arrows.

## Notes

- The dye segmentation relies on the saturation channel of the HSV color space and Otsu thresholding. For different lighting conditions you may need to adjust the morphological kernel size or post-processing steps.
- If the red markers are not detected, tweak the thresholds in `detect_red_markers` or adjust `min_area`.
- For the OCR step, try increasing the `--digits-width`/`--digits-height` ratios if multiple numbers are present in the bottom-right area.
