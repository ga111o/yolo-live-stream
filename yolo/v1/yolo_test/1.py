import os
import random
import shutil  # 파일 복사를 위해 추가
from ultralytics import YOLO

# 학습 이미지 경로와 라벨 정의


train_images_path = './yolo/train_images'
labels = ['알콜 솜']

# YOLO 모델 인스턴스 생성
model = YOLO('yolov8n.pt')  # YOLOv8 nano 모델 사용

# 학습을 위한 데이터셋 준비 함수
def prepare_dataset(image_path, label):
    dataset_path = 'yolo_dataset'
    os.makedirs(dataset_path, exist_ok=True)

    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for image_file in os.listdir(image_path):
        if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # 이미지 확장자 추가
            # 이미지 복사 (os.system 대신 shutil.copy 사용)
            src_image_path = os.path.join(image_path, image_file)
            dst_image_path = os.path.join(images_dir, image_file)
            shutil.copy(src_image_path, dst_image_path)
            print(f"Copied {src_image_path} to {dst_image_path}")  # 디버깅을 위한 출력

            # 라벨 파일 생성
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'w') as f:
                # 랜덤한 위치와 크기로 바운딩 박스 생성
                center_x = random.uniform(0.3, 0.7)
                center_y = random.uniform(0.3, 0.7)
                width = random.uniform(0.2, 0.6)
                height = random.uniform(0.2, 0.6)
                f.write(f'0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n')
            print(f"Created label file: {label_path}")  # 디버깅을 위한 출력

    return dataset_path

# 데이터셋 설정 파일 생성
def create_dataset_yaml(dataset_path):
    yaml_content = f"""
path: {os.path.abspath(dataset_path)}
train: images
val: images

names:
  0: 알콜 솜
"""
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

# 데이터셋 준비
dataset_path = prepare_dataset(train_images_path, labels)
yaml_path = create_dataset_yaml(dataset_path)

# YOLO 모델 학습 (yaml 파일 경로 사용)
model.train(data=yaml_path, epochs=10, imgsz=640)

# 학습된 모델 저장
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f'./trained_model_{current_time}.pt')
