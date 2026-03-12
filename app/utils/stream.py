import asyncio
from vidgear.gears import CamGear
from typing import AsyncGenerator
import numpy as np

class StreamGenerator:
    def __init__(self, url: str, resolution: str = "720p"):
        self.url = url
        self.options = {"STREAM_RESOLUTION": resolution}
        self.stream: CamGear | None = None

    def _start_stream(self):
        if self.stream is None:
            self.stream = CamGear(
                source=self.url, 
                stream_mode=True, 
                logging=False, 
                **self.options
            ).start()

    async def read_stream(self) -> AsyncGenerator[np.ndarray, None]:
        if self.stream is None:
            self._start_stream()

        try:
            while True:
                frame = await asyncio.to_thread(self.stream.read)

                if frame is None:
                    break

                yield frame
                await asyncio.sleep(0) 

        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream = None