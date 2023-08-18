from fastapi import FastAPI
import app.Routers.Sign as Sign
import app.Routers.Chatbot as Chatbot
import app.Routers.ChatLog as ChatLog
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import app.Utils.pinecone as pc

# pc.train_text()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(Sign.router, tags=["sign"], prefix="/auth")
app.include_router(Chatbot.router, tags=["Chatbot"])
app.include_router(ChatLog.router, tags=["ChatLog"])


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7000, reload=True)
