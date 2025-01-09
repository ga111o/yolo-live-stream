import { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [detections, setDetections] = useState([]);
  const videoRef = useRef(null);

  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("cam:", err);
      }
    };
    setupCamera();

    let ws = null;
    let animationFrameId = null;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket("ws://localhost:8000/ws");

        ws.onopen = () => {
          console.log("WebSocket Connected");
          // 연결되면 프레임 전송 시작
          sendFrame();
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.detections && data.detections.length > 0) {
              const highestConfidenceDetection = data.detections.reduce(
                (prev, current) => {
                  return prev.confidence > current.confidence ? prev : current;
                }
              );
              console.log(
                `${highestConfidenceDetection.class} (${(
                  highestConfidenceDetection.confidence * 100
                ).toFixed(1)}%)`
              );
            }
          } catch (error) {
            console.error("Error parsing WebSocket message:", error);
          }
        };

        ws.onerror = (error) => {
          console.error("WebSocket error:", error);
        };

        ws.onclose = (e) => {
          console.log(
            "Socket is closed. Reconnect will be attempted in 1 second.",
            e.reason
          );
          setTimeout(connectWebSocket, 1000);
        };
      } catch (error) {
        console.error("Error creating WebSocket connection:", error);
        setTimeout(connectWebSocket, 1000);
      }
    };

    const sendFrame = () => {
      if (ws?.readyState === WebSocket.OPEN && videoRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              ws.send(blob);
            }
          },
          "image/jpeg",
          0.8
        );
      }
      animationFrameId = setTimeout(sendFrame, 5000);
    };

    connectWebSocket();

    return () => {
      if (animationFrameId) {
        clearTimeout(animationFrameId);
      }
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const getBoundingBoxStyle = (bbox) => {
    return {
      position: "absolute",
      left: `${bbox[0]}px`,
      top: `${bbox[1]}px`,
      width: `${bbox[2] - bbox[0]}px`,
      height: `${bbox[3] - bbox[1]}px`,
      border: "2px solid red",
      color: "red",
      backgroundColor: "rgba(255, 0, 0, 0.1)",
    };
  };

  return (
    <div>
      <div style={{ position: "relative" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ maxWidth: "100%" }}
        />
      </div>
    </div>
  );
}

export default App;
