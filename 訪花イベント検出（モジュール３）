#!/usr/bin/env python3
import os
import csv
import time
import cv2
import numpy as np
from datetime import datetime

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# =========================
# Paths
# =========================
BASE_DIR = os.path.expanduser("~/visit_detect")
SAVE_DIR = os.path.join(BASE_DIR, "events")
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "event_log.csv")

# =========================
# Camera settings
# =========================
LORES_SIZE = (640, 480)      # lightweight detection stream
MAIN_SIZE = (1280, 720)      # saved video size
FPS = 20.0
RECORD_SECONDS = 10          # 7–8秒平均を考えて少し長め
COOLDOWN_SECONDS = 3

# =========================
# ROI (x1, y1, x2, y2)
# Adjust this to the flower position in the 640x480 detection view
# =========================
ROI = (240, 140, 420, 320)

# =========================
# Motion detection parameters
# =========================
BLUR_SIZE = (9, 9)
DIFF_THRESHOLD = 22

# lower: ignore tiny noise
# upper: ignore very large wind-driven movement
MIN_CHANGED_AREA = 120
MAX_CHANGED_AREA = 3500

# trigger only after N consecutive positive frames
CONSECUTIVE_FRAMES_REQUIRED = 3

# morphology
OPEN_KERNEL = 3
DILATE_ITER = 2

# save latest debug image
SAVE_DEBUG_LATEST = True

# =========================
# Rough size classes based on largest contour area
# =========================
SIZE_THRESHOLDS = {
    "small_max": 500,
    "medium_max": 1500
}

# H.264 bitrate
BITRATE = 10_000_000


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "video_file",
                "changed_area",
                "largest_contour_area",
                "size_class",
                "roi_x1",
                "roi_y1",
                "roi_x2",
                "roi_y2"
            ])


def append_csv(row):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def clamp_roi(roi, width, height):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, width - 1))
    x2 = max(1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(1, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def preprocess(gray, roi):
    x1, y1, x2, y2 = roi
    crop = gray[y1:y2, x1:x2]
    crop = cv2.GaussianBlur(crop, BLUR_SIZE, 0)
    return crop


def classify_size(largest_contour_area):
    if largest_contour_area <= SIZE_THRESHOLDS["small_max"]:
        return "small"
    elif largest_contour_area <= SIZE_THRESHOLDS["medium_max"]:
        return "medium"
    else:
        return "large"


def main():
    ensure_csv_header()

    picam2 = Picamera2()

    # Preview window:
    # works only in local desktop / VNC desktop session, not plain SSH/Termius
    picam2.start_preview(Preview.QT)

    config = picam2.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "YUV420"},
        controls={"FrameRate": FPS},
        buffer_count=6,
        queue=False
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    roi = clamp_roi(ROI, LORES_SIZE[0], LORES_SIZE[1])

    print("Visit detector started")
    print(f"Saving events to: {SAVE_DIR}")
    print(f"CSV log: {CSV_PATH}")
    print(f"ROI: {roi}")
    print("Preview window should be visible on desktop/VNC")
    print("Press Ctrl+C to stop")

    prev_proc = None
    cooldown_until = 0
    positive_count = 0

    try:
        while True:
            lores = picam2.capture_array("lores")
            if lores is None:
                continue

            if len(lores.shape) == 2:
                gray = lores
            else:
                gray = cv2.cvtColor(lores, cv2.COLOR_BGR2GRAY)

            current_proc = preprocess(gray, roi)

            if prev_proc is None:
                prev_proc = current_proc
                continue

            diff = cv2.absdiff(prev_proc, current_proc)
            _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, kernel, iterations=DILATE_ITER)

            changed_area = int(np.count_nonzero(thresh))

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            largest_contour_area = 0.0
            if contours:
                largest_contour_area = max(cv2.contourArea(c) for c in contours)

            valid_motion = (
                changed_area >= MIN_CHANGED_AREA and
                changed_area <= MAX_CHANGED_AREA
            )

            if valid_motion:
                positive_count += 1
            else:
                positive_count = 0

            if SAVE_DEBUG_LATEST:
                dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                x1, y1, x2, y2 = roi
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    dbg,
                    f"changed={changed_area} largest={int(largest_contour_area)} pos={positive_count}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(os.path.join(BASE_DIR, "debug_latest.jpg"), dbg)

            now = time.time()
            triggered = (
                positive_count >= CONSECUTIVE_FRAMES_REQUIRED and
                now >= cooldown_until
            )

            if triggered:
                ts = now_str()
                size_class = classify_size(largest_contour_area)
                out_path = os.path.join(SAVE_DIR, f"visit_{ts}_{size_class}.h264")

                print(
                    f"[TRIGGER] changed={changed_area}, "
                    f"largest={largest_contour_area:.1f}, size={size_class}"
                )
                print(f"[RECORDING] {out_path}")

                encoder = H264Encoder(BITRATE)
                picam2.start_encoder(encoder, FileOutput(out_path))
                time.sleep(RECORD_SECONDS)
                picam2.stop_encoder()

                print(f"[SAVED] {out_path}")

                append_csv([
                    ts,
                    out_path,
                    changed_area,
                    round(largest_contour_area, 1),
                    size_class,
                    roi[0], roi[1], roi[2], roi[3]
                ])

                cooldown_until = time.time() + COOLDOWN_SECONDS
                positive_count = 0

            prev_proc = current_proc

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        picam2.stop_preview()
        picam2.stop()


if __name__ == "__main__":
    main()
