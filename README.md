# Dye Flow Image Analysis

This repository contains a Python script that performs a computer-vision analysis of dye-flow imagery sourced from an OBS video stream (or captured frames saved to disk).

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

Run the analysis from the repository root, pointing the script at the OBS stream URL (such as an RTMP endpoint) or the virtual camera device that OBS exposes:

```bash
python process_image.py --stream-url rtmp://127.0.0.1:1935/live/experiment
```

If the stream is continuous, you can limit the processing time by specifying `--frame-limit` so that only the first *N* frames are consumed. The script retries fetching the initial frame up to 50 times with a 0.1 second delay; adjust this behaviour with `--connect-retries` and `--retry-delay` if OBS needs extra time to start broadcasting.

Key command line options:

- `--stream-url`: OBS stream URL or device identifier that OpenCV can open.
- `--output`: Directory where results are stored (default: `outputs/`).
- `--padding`: Extra pixels to add around the detected markers when cropping.
- `--digits-width`, `--digits-height`: Ratios that control how much of the bottom-right corner is used for OCR. Increase them if the numbers are not fully captured.
- `--frame-limit`: Maximum number of frames to read from the stream (0 means read until the stream ends).
- `--connect-retries` / `--retry-delay`: Control how many times and how often the script retries reading the first frame.

All generated artifacts are written to the output directory:

- `cropped_region.png`: Color crop between the detected markers in the primary image.
- `marker_detection.png`: Visualization of the detected marker centers.
- `dye_mask.png`: Binary mask showing the detected dye.
- `dye_overlay.png`: Original image with the dye mask overlaid for quick inspection.
- `digits_region.png`: The bottom-right crop used for OCR.
- `numeric_readings.csv`: CSV containing the recognized numbers, OCR confidence, and bounding boxes in the original image coordinates.
- `summary.txt`: Text file summarizing the area statistics and OCR results.
- `cropped_region_second.png`: Matched crop from the secondary image used for optical flow.
- `cropped_region_mask.png`: Cropped dye mask corresponding to the marker region.
- `sparse_flow_vectors.csv`: Sparse optical flow vectors stored as float32 values in `[x, y, delta_x, delta_y]` format.
- `sparse_flow_visualization.png`: Reference crop overlaid with the sparse vector field arrows.

## Notes

- The dye segmentation relies on the saturation channel of the HSV color space and Otsu thresholding. For different lighting conditions you may need to adjust the morphological kernel size or post-processing steps.
- If the red markers are not detected, tweak the thresholds in `detect_red_markers` or adjust `min_area`.
- For the OCR step, try increasing the `--digits-width`/`--digits-height` ratios if multiple numbers are present in the bottom-right area.
