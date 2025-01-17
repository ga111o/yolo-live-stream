from ultralytics import YOLO
import os
from pathlib import Path

def predict_images():
    # 학습된 모델 로드
    model = YOLO('results.pt')
    
    # 테스트 이미지 디렉토리 설정
    test_dir = 'test'
    
    # 결과를 저장할 디렉토리 생성
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 테스트 디렉토리의 모든 이미지에 대해 추론 실행
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, img_file)
            
            # 이미지에 대한 예측 수행
            results = model.predict(img_path, save=True, project=results_dir)
            
            # 예측 결과 출력
            for result in results:
                print(f"\nPredictions for {img_file}:")
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    print(f"- Class: {class_name}, Confidence: {confidence:.2f}")

def main():
    try:
        predict_images()
        print("Prediction completed! Check 'test_results' directory for visualized results.")
    except Exception as e:
        print(f"---error---\n{str(e)}")

if __name__ == "__main__":
    main()
