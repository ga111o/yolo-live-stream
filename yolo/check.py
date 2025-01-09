import torch
from ultralytics import YOLO
import os
from PIL import Image

# 학습된 모델 로드
model = YOLO('./trained_model_20250109_182911.pt')

# train_images 디렉토리의 모든 이미지 처리
image_dir = './yolo/train_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 모든 이미지에 대해 반복
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    # 이미지 로드 및 예측
    results = model.predict(image_path, save=True, conf=0.25)  # conf는 confidence threshold
    
    # 결과 출력
    # print(f"\n=================\n처리중인 이미지: {image_file}")
    # print("예측 결과:")
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         # 바운딩 박스 좌표
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         # 신뢰도
    #         confidence = box.conf[0]
    #         # 클래스
    #         cls = box.cls[0]
    #         class_name = model.names[int(cls)]
            
    #         print(f"클래스: {class_name}, 신뢰도: {confidence:.2f}")
    # print("=================")
    # print("=================\n")
