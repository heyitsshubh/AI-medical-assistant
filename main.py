from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables early so route modules can read them on import
load_dotenv()

from server.middlewares.exception_handlers import catch_exception_middleware
from server.routes.upload_pdfs import router as upload_router
from server.routes.ask_question import router as ask_router


app = FastAPI(
    title="Medical Assistant API",
    description="API for AI Medical Assistant Chatbot"
)


# Exception Middleware
app.middleware("http")(catch_exception_middleware)


# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "message": "Medical AI API Running"
    }


# 1. upload pdf documents
app.include_router(upload_router)


# 2. asking query
app.include_router(ask_router)  