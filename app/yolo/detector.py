import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from app.settings import Settings

class CarDetector:
    model: YOLO = YOLO("app/models/yolo11n.pt")

    async def predict(self, frame: np.ndarray) -> list[Results]:
        return self.model.predict(source=frame, classes=[2], device=Settings.CUDA, verbose=False, conf=0.35)