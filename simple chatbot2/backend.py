from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

print(os.getcwd())

load_dotenv()

with open("prompt.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(api_key=os.getenv("API_KEY"))


@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'frontend.html')


def classify(message):
    prompt = PROMPTS["classification_prompt"].format(message=message)

    classification = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    return classification.output_text.strip()



def generate_chat_response(user_message, category):

    system_prompt = PROMPTS["system_prompt"].format(category=category)

    final_prompt = f"""
    {system_prompt}

    پیام کاربر:
    {user_message}

    پاسخ دقیق و مرتبط تولید کن.
    """

    response = client.responses.create(
        model="gpt-5-nano",
        input=final_prompt
    )

    return response.output_text.strip()



@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'error': 'پیام خالی است'}), 400

    category = classify(user_message)
    print("CATEGORY:", category)

    reply = generate_chat_response(user_message, category)

    return jsonify({'reply': reply})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

