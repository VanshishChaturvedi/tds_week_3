import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str

@app.post("/answer-image", response_model=QAResponse)
def answer_image(payload: QARequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Render")

    # Clean the base64 string (strip data URI if present)
    b64_str = payload.image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]

    system_instruction = (
        "You are an expert document analysis assistant. Answer the user's question accurately based "
        "strictly on the provided document image.\n"
        "CRITICAL FORMATTING RULES:\n"
        "- For numeric answers, return ONLY the raw numeric digits (e.g., '4089.35').\n"
        "- Do NOT include currency symbols, commas as unit marks, or text units.\n"
        "- Strip all conversational filler text."
    )

    url = "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # REST API structure for multimodal (image + text)
    payload_data = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": b64_str}},
                    {"text": payload.question}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload_data)
        response.raise_for_status() 
        
        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        return QAResponse(answer=raw_text)
        
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, requests.exceptions.HTTPError):
            error_msg += f" - Response Text: {response.text}"
        print(f"CRITICAL INFERENCE ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
