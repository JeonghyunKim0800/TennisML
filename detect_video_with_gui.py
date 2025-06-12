import torch
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import numpy as np

# === GUI CONFIGURATION ===
def gui_config():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    print("Select the video file:")
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if not video_path:
        print("No video selected. Exiting...")
        exit()

    print("Select the output directory:")
    output_dir = filedialog.askdirectory()
    if not output_dir:
        print("No output directory selected. Exiting...")
        exit()

    model_path = 'yolov8x.pt'  # Predefined model path

    return video_path, model_path, output_dir

# === MAIN FUNCTION ===
def main():
    video_path, model_path, output_dir = gui_config()

    # Set up paths
    output_video_path = os.path.join(output_dir, "output_yolov8x.mp4")
    frame_output_dir = os.path.join(output_dir, "images/train")
    label_output_dir = os.path.join(output_dir, "labels/train")

    # Create directories
    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Load YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)

    # Video setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0

    # Process frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame for dataset
        frame_filename = os.path.join(frame_output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Inference
        results = model.predict(frame, device=device, conf=0.15, classes=[0, 32])
        boxes = results[0].boxes

        label_filename = os.path.join(label_output_dir, f"frame_{frame_idx:05d}.txt")
        with open(label_filename, 'w') as f:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                if cls not in [0, 32]:
                    continue
                # Convert to YOLO format
                x1, y1, x2, y2 = xyxy
                cx = ((x1 + x2) / 2) / width
                cy = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Draw boxes
        for box in boxes:
            xyxy = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{cls} {conf:.2f}'
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Detection complete. Output video saved to {output_video_path}")

    # Create tennis.yaml
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/train  # Same as train for now
names:
  0: person
  32: sports ball
"""
    yaml_path = os.path.join(output_dir, "tennis.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Dataset and labels saved. YAML configuration saved to {yaml_path}")

if __name__ == "__main__":
    main()
