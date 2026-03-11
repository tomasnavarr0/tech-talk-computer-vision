import cv2
import torch
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from vidgear.gears import CamGear
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# 1. Configuración de Hardware y Modelo
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)

# 2. Configuración del Stream
youtube_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
options = {"STREAM_RESOLUTION": "720p"}
stream = CamGear(source=youtube_url, stream_mode=True, logging=False, **options).start()

def generate_frames():
    while True:
        frame = stream.read()
        if frame is None:
            break

        # Inferencia de detección de autos (class 2)
        results = model.predict(source=frame, classes=[2], device=device, verbose=False, conf=0.35)
        
        # Dibujar boxes
        annotated_frame = results[0].plot()

        # Codificar el frame en JPEG para enviarlo por la web
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Formato MJPEG para el navegador
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def index():
    return Response(content="<h1>Streaming AI Demo</h1><img src='/video'>", media_type="text/html")

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print(f"🚀 Servidor iniciado en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)