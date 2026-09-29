import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None
SYSTEM_PROMPT = (BASE_DIR / "chatbot_config.txt").read_text(encoding="utf-8")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400
    if len(message) > 6000:
        return jsonify({"error": "Please keep your question under 6,000 characters."}), 413
    if client is None:
        return jsonify({"error": "Gemini API key is missing. Add it to the .env file."}), 503

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=1200,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn’t create an answer just now. Please try again."
        return jsonify({"answer": answer})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "I couldn’t reach the AI service. Please check your API key and try again."
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
