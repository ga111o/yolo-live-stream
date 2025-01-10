import torch
from ultralytics import YOLO
import os
from PIL import Image
import time
from icecream import ic

# 학습된 모델 로드
model = YOLO('./trained_model_20250109_182911.pt')

# 예측 시간 측정 시작
start_time = time.time()

# 전체 디렉토리에 대한 예측 수행
results = model.predict(source='./yolo/test_images', save=True, conf=0.25)

# 예측 시간 측정 종료
end_time = time.time()
prediction_time_ms = (end_time - start_time) * 1000
prediction_time_ms = f"{prediction_time_ms:.3f}"

# 전체 예측 시간 출력
ic(prediction_time_ms)

# 결과 처리
for r in results:
    # 이미지 파일 이름 출력
    ic(os.path.basename(r.path))
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
    