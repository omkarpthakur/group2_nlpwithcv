import cv2
import os
import shutil
from tqdm import tqdm

input_dir = 'videos'
output_dir = 'frame_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

for video_file in video_files:
    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(input_dir, video_file)

    video_output_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit=" frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = os.path.join(video_output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_file, frame)

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()

print("Video frames extraction and storage completed.")
