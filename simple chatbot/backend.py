from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(api_key="")

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'frontend.html')

def classify(message):
    prompt = f"""
    پیام زیر را از نظر نوع دسته‌بندی کن. فقط یکی از این موارد را برگردان:
    - پرسش عمومی
    - تشخیص بیماری
    -پیدا کردن بیمارستان
    -رزرو وقت
    -پیدا کردن داروخانه و دارو

    پیام:
    {message}

    فقط نوع پیام را برگردان.
    """

    classification = client.responses.create(
        model="gpt-5-nano", 
        input=prompt
    )

    return classification.output_text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    print(classify(user_message))
    if not user_message:
        return jsonify({'error': 'پیام خالی است'}), 400

    response = client.responses.create(
        model="gpt-5-nano",
        input=user_message
    )
    return jsonify({'reply': response.output_text.strip()})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

