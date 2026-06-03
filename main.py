from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.middlewares.exception_handlers import catch_exception_middleware



app=FastAPI(title="Medical Assistant API",description="API for AI Medical Assistant Chatbot")
app.add_middleware(catch_exception_middleware)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
