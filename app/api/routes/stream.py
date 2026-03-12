from fastapi import APIRouter, Response, Depends
from fastapi.responses import StreamingResponse
from app.processors.car import CarProcessor

router = APIRouter(prefix="/v1/stream", tags=["Stream"])

def get_car_processor() -> CarProcessor:
    return CarProcessor()

@router.get("/")
async def index():
    content = """
    <html>
        <head><title>Streaming AI Demo</title></head>
        <body>
            <h1>Streaming AI Demo</h1>
            <img src="video" width="100%">
        </body>
    </html>
    """
    return Response(content=content, media_type="text/html")

@router.get("/video")
async def video_feed(processor: CarProcessor = Depends(get_car_processor)):
    return StreamingResponse(
        processor.process_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )