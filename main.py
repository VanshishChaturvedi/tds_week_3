import os
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# JSON Note: "Enable CORS: the grader sends requests from a Cloudflare Worker."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
client = genai.Client(api_key=api_key)

# JSON Spec: request_body
class QARequest(BaseModel):
    image_base64: str
    question: str

# JSON Spec: response_body
class QAResponse(BaseModel):
    answer: str

# JSON Spec: method="POST", path="/answer-image"
@app.post("/answer-image", response_model=QAResponse)
async def process_image_qa(payload: QARequest):
    try:
        # Clean the base64 string just in case it has a data URI header
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
            
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    # JSON Spec Notes: "The 'answer' field must always be a string. For numeric answers, return the number as a string (e.g. '4089.35'). Do not include units or extra text"
    system_instruction = (
        "You are an API data extractor. Answer the question based on the image.\n"
        "RULES:\n"
        "1. Return ONLY the answer. No conversational text.\n"
        "2. If the answer is a number, return ONLY the raw digits and decimal points (e.g., '4089.35').\n"
        "3. NEVER include units, currency symbols ($, Rs), or words like 'total'."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                payload.question
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        
        # Ensure it returns as a string exactly formatted as {"answer": "..."}
        final_answer = str(response.text).strip() if response.text else ""
        return QAResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
