import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.api.routes.stream import router as stream_router

app = FastAPI(title="Vehicle Detection Stream")


app.include_router(stream_router)

@app.get("/")
async def root():
    return RedirectResponse(url="/v1/stream/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)