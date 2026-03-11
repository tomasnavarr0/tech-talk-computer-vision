import cv2
import torch
from vidgear.gears import CamGear
from ultralytics import YOLO

def main() -> None:
    # Verificamos si CUDA está disponible para mostrar el uso de la 5060 en la charla
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Dispositivo: {torch.cuda.get_device_name(0) if device == '0' else 'CPU'}")

    # 1. Cargamos el modelo YOLO11n (Nano) y lo movemos a la GPU
    model = YOLO("yolo11n.pt").to(f"cuda:{device}")

    # 2. Configuramos CamGear para el stream
    # Para autos, quizás una cámara de tráfico sea mejor (ejemplo: Times Square o alguna autopista)
    youtube_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk" 
    
    options = {"STREAM_RESOLUTION": "720p"} 
    stream = CamGear(
        source=youtube_url,
        stream_mode=True,
        logging=False,
        **options
    ).start()

    print("🚗 Detección de Vehículos Iniciada. Presiona 'q' para salir.")

    while True:
        frame = stream.read()
        if frame is None:
            break

        # 3. Inferencia de detección (Predict)
        # classes=[2]: Filtramos solo para 'car'
        # Si quieres detectar camiones (7) o buses (5), puedes poner classes=[2, 5, 7]
        results = model.predict(
            source=frame,
            classes=[2],  # ID 2 es 'car' en el dataset COCO
            device=device,
            verbose=False,
            conf=0.35      # Confianza un poco más alta para evitar falsos positivos
        )

        # 4. Dibujamos los resultados en el frame
        # Usamos results[0] porque le pasamos un solo frame
        annotated_frame = results[0].plot()
        
        # 5. Visualización
        cv2.imshow("CV Demo - Detección de Autos en Tiempo Real", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    main()