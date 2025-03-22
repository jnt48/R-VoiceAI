from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API with your key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise Exception("Gemini API key not set. Please set GEMINI_API_KEY in your environment variables.")
genai.configure(api_key=gemini_api_key)

# Initialize FastAPI app
app = FastAPI(title="EDIT Chatbot API")

# Enable CORS to allow frontend to interact with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Data models for the conversation
class ChatMessage(BaseModel):
    role: str  # "user" or "bot"
    message: str

class ChatRequest(BaseModel):
    user_message: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    bot_message: str
    history: List[ChatMessage]

# Chatbot class that uses Gemini AI
class Chatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def get_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content([prompt])
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# Create an instance of the chatbot
chatbot = Chatbot()

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_request: ChatRequest):
    # Get the conversation history or initialize empty
    history = chat_request.history or []
    
    # Append the new user message to the history
    history.append(ChatMessage(role="user", message=chat_request.user_message))
    
    # Limit history to the last 100 messages
    if len(history) > 100:
        history = history[-100:]
    
    # Build a conversation prompt with a system instruction
    conversation_prompt = (
        "System: You are EDIT, an educational, professional, interactive, and caring chatbot. "
        "Your responses should be clear, detailed, and supportive. You provide thoughtful, context-aware answers "
        "and help address any worries the user may have. Always maintain a professional tone while being empathetic.\n"
    )
    for chat in history:
        if chat.role == "user":
            conversation_prompt += "User: " + chat.message + "\n"
        else:
            conversation_prompt += "EDIT: " + chat.message + "\n"
    conversation_prompt += "EDIT: "  # Prompt the next answer
    
    # Get the bot's response from Gemini AI
    bot_message = chatbot.get_response(conversation_prompt)
    
    # Append the bot response to the conversation history
    history.append(ChatMessage(role="bot", message=bot_message))
    
    # Return the bot's response and updated history
    return ChatResponse(bot_message=bot_message, history=history)

