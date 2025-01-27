import os

IMAGE_DIR = "C:/image_data/image_data/"
LOG_FILE = "C:/logging_data/log_file_4.txt"

missing_files = []

with open(LOG_FILE, 'r') as file:
    for line in file:
        parts = line.strip().split()
        front_frame = f"{parts[3]}.jpg"  # 第四列是 front_frame
        left_frame = f"{parts[1]}.jpg"  # 第二列是 left_frame
        right_frame = f"{parts[2]}.jpg"  # 第三列是 right_frame
        
        # 检查图片文件是否存在
        for frame in [front_frame, left_frame, right_frame]:
            full_path = os.path.join(IMAGE_DIR, frame)
            print(f"Checking: {full_path}")  
            if not os.path.exists(full_path):
                missing_files.append(full_path)

if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print("No missing files detected.")
