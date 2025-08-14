from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

chat_history = []

app = Flask(__name__)
CORS(app)

class ChatRequest(BaseModel):
    question: str

@app.route("/query", methods=["POST"])
def query_chatbot():
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Question can't be empty"}), 400
    if len(question) > 500:
        return jsonify({"error": "Question too long (max 500 characters)"}), 400

    try:
        # Fixed: Use chr(10) for newline instead of \n in the f-string
        chat_history_str = "".join(f"Human: {entry['question']}{chr(10)}AI: {entry['answer']}{chr(10)}" for entry in chat_history)
        
        prompt = f"""
You are an FAQ chatbot for NirveonX, an AI-powered health and wellness platform.
Use this info to provide accurate, concise, and helpful answers: NirveonX works on web browsers,
iOS, and Android, with plans to support smart wearables and health devices for better tracking.
We have a free basic plan with core AI health insights, plus premium plans with extra features and support.
Check our website for pricing details.
Your data is safe with us—we use top-notch encryption, GDPR-compliant practices, and secure cloud storage, and we never share it without your consent. 
Our AI uses natural language processing and machine learning to give personalized health and wellness tips, learning from research and user data. 
Our health insights are accurate, based on vetted algorithms, but we're not doctors. 
If the question isn't covered, use this info to give a relevant answer.
Keep it short, friendly, and professional. Use chat history for context.

Chat History:
{chat_history_str}

Question: {question}

Answer:
"""
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # or another Groq-supported model
            messages=[
                {"role": "system", "content": "You're a NirveonX FAQ assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )

        answer = response.choices[0].message.content.strip()

        chat_history.append({"question": question, "answer": answer})

        if len(chat_history) > 10:
            chat_history.pop(0)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

@app.route("/")
def root():
    return redirect("/docs")

if __name__ == "__main__":
    app.run(debug="true", host="0.0.0.0", port=5000)