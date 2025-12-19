from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import json, os, base64
from diagnosing.radiology.breakpoints.train import predict as predict_fracture
from diagnosing.radiology.chest.run import predict as predict_chest
from diagnosing.radiology.teeth.ZB import predict as predict_dental

load_dotenv()

with open("prompt.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("API_KEY"))

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'frontend.html')


def classify_message(message):
    prompt = PROMPTS["classification_prompt"].format(message=message)
    classification = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return classification.output_text.strip()


def generate_chat_response(message, category):
    system_prompt = PROMPTS["system_prompt"].format(category=category)
    final_prompt = f"""
{system_prompt}

پیام کاربر:
{message}

پاسخ دقیق و مرتبط تولید کن.
"""
    response = client.responses.create(
        model="gpt-5-nano",
        input=final_prompt
    )
    return response.output_text.strip()


def explain_with_ai(result, category):
    prompt = f"""
نتیجه تحلیل تصویر پزشکی:

نوع تصویر: {category}
تشخیص: {result['diagnosis']}
میزان اطمینان: {result['confidence']}

به زبان ساده توضیح بده:
- آیا مشکل جدی است؟
- آیا نیاز به مراجعه به پزشک هست؟
- لحن آرام و غیرترسناک داشته باش
- متن نهایتا 3 خط
- هیچگونه توضیح پزشکی نکن
- در پایان هیچ سوالی نپرس
"""
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return response.output_text.strip()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()
    image_data = data.get('image', None)

    category = classify_message(message).lower() if message else "تشخیص بیماری"

    if category == "پرسش عمومی" and not image_data:
        reply = generate_chat_response(message, category)
        return jsonify({"reply": reply})

    image_path = None
    if image_data:
        filename = image_data.get('filename', 'upload.png')
        content = base64.b64decode(image_data.get('content'))
        image_path = os.path.join(UPLOAD_DIR, filename)
        with open(image_path, "wb") as f:
            f.write(content)

    if message:
        msg_lower = message.lower()
        if "دندان" in msg_lower or "teeth" in msg_lower:
            result = predict_dental(image_path)
            category_name = "dental"
        elif "شکستگی" in msg_lower or "fracture" in msg_lower:
            result = predict_fracture(image_path)
            category_name = "fracture"
        elif "قفسه سینه" in msg_lower or "ریه" in msg_lower or "chest" in msg_lower:
            result = predict_chest(image_path)
            category_name = "chest"
        else:
            return jsonify({"reply": "لطفاً نوع تصویر (دندان، شکستگی، قفسه سینه) را مشخص کنید"}), 400
    else:
        return jsonify({"reply": "پیام متنی لازم است"}), 400

    reply = explain_with_ai(result, category_name)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)













