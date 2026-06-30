import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """You are an expert agricultural assistant helping farmers.
Be clear, practical, and avoid jargon. When given a disease diagnosis,
explain it simply and give organic + chemical treatment options,
fertilizer advice, and prevention tips. Keep responses concise and actionable."""


def build_diagnosis_prompt(crop: str, disease: str, confidence: float) -> str:
    if disease.lower() == "healthy":
        return f"The uploaded {crop} plant appears healthy with {confidence}% confidence. Give the farmer brief preventive care tips."
    
    return f"""The uploaded image shows {crop} affected by {disease} with {confidence}% confidence.

Explain in simple language:
- Probable causes
- Symptoms to watch for
- Organic treatment
- Chemical treatment
- Fertilizer recommendations
- Preventive measures"""


def call_groq(prompt: str, history: list = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message.content


def call_gemini(prompt: str, history: list = None) -> str:
    convo_text = SYSTEM_PROMPT + "\n\n"
    if history:
        for h in history:
            role = "Farmer" if h["role"] == "user" else "Assistant"
            convo_text += f"{role}: {h['content']}\n"
    convo_text += f"Farmer: {prompt}"

    response = gemini_model.generate_content(convo_text)
    return response.text


def generate_response(prompt: str, history: list = None) -> dict:
    """Try Groq first, fallback to Gemini on failure."""
    try:
        text = call_groq(prompt, history)
        return {"text": text, "source": "groq"}
    except Exception as e:
        print(f"Groq failed: {e}")
        try:
            text = call_gemini(prompt, history)
            return {"text": text, "source": "gemini"}
        except Exception as e2:
            print(f"Gemini also failed: {e2}")
            return {
                "text": "Sorry, I'm having trouble connecting right now. Please try again in a moment.",
                "source": "none"
            }