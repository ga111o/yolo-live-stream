from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.websockets import WebSocket
import asyncio

app = FastAPI()



origins = [
    "http://localhost:5173",
    "http://localhost:5173/",
    "http://localhost:3000",
    "http://localhost:3000/",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173/",
    "http://127.0.0.1:3000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('./trained_model.pt')


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        while True:
            # 웹소켓으로부터 이미지 데이터 수신
            data = await websocket.receive_bytes()
            
            # bytes를 numpy array로 변환
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # YOLO로 객체 감지
            results = model(frame)
            
            # 감지된 객체 정보 추출 및 전송
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    name = model.names[int(cls)]
                    detections.append({
                        "class": name,
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            # 웹소켓으로 결과 전송
            await websocket.send_json({"detections": detections})
            
            # 5초 대기
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
