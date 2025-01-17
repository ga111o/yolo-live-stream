import torch
from ultralytics import YOLO
import os
from PIL import Image
import time
from icecream import ic

# 학습된 모델 로드
model = YOLO('./trained_model_20250109_182911.pt')

# train_images 디렉토리의 모든 이미지 처리
image_dir = './yolo/test_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 모든 이미지에 대해 반복
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    start_time = time.time()
    
    results = model.predict(image_path)
    
    end_time = time.time()
    prediction_time_ms = (end_time - start_time) * 1000
    prediction_time_ms = f"{prediction_time_ms:.3f}"
    
    # 결과 출력
    ic(image_file, prediction_time_ms)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0]
            # 신뢰도
            confidence = box.conf[0]
            # 클래스
            cls = box.cls[0]
            class_name = model.names[int(cls)]
            
            ic(class_name, confidence)
    