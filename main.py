from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import predict_disease
from schemas import PredictionResponse
from llm_service import generate_response, build_diagnosis_prompt
from schemas import ChatRequest, ChatResponse

app = FastAPI(title="Bhoomi_AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")
    
    image_bytes = await file.read()
    result = predict_disease(image_bytes)
    
    return {**result, "status": "success"}


@app.post("/diagnose-explain", response_model=ChatResponse)
async def diagnose_explain(file: UploadFile = File(...)):
    """Combines Module 1 prediction + Module 2 LLM explanation in one call."""
    image_bytes = await file.read()
    prediction = predict_disease(image_bytes)
    
    prompt = build_diagnosis_prompt(
        prediction["crop"], prediction["disease"], prediction["confidence"]
    )
    result = generate_response(prompt)
    
    return {"response": result["text"], "source": result["source"]}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Follow-up questions, with optional crop/disease context."""
    history = [h.model_dump() for h in req.history] if req.history else []

    prompt = req.message
    if req.crop and req.disease:
        prompt = f"(Context: {req.crop} with {req.disease}) {req.message}"

    result = generate_response(prompt, history)

    return {
        "response": result["text"],
        "source": result["source"]
    }