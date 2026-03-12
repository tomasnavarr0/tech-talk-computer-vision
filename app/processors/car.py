from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import cv2

from app.yolo.detector import CarDetector
from app.utils.stream import StreamGenerator
from app.settings import Settings

@dataclass
class CarProcessor:
    car_detector: CarDetector = field(default_factory=CarDetector)

    async def process_stream(self) -> AsyncGenerator[bytes, None]:
        stream_generator = StreamGenerator(Settings.STREAM_URL)
        async for frame in stream_generator.read_stream():
            results = await self.car_detector.predict(frame)

            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                raise BufferError("No frame to continue")
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
