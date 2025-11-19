from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(api_key=os.getenv("API_KEY"))


@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'frontend.html')


def classify(message):
    prompt = f"""
    تو یک طبقه‌بند پیام هستی.
    فقط و فقط یکی از این دسته‌ها را خروجی بده و هیچ متن دیگری تولید نکن:

    - پرسش عمومی
    - تشخیص بیماری
    - پیدا کردن بیمارستان
    - رزرو وقت
    - پیدا کردن داروخانه و دارو

    پیام:
    {message}

    خروجی فقط نام دسته باشد. هیچ توضیحی اضافه نده.
    """

    classification = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    return classification.output_text.strip()


def generate_chat_response(user_message, category):

    system_prompt = f"""
    تو یک دستیار پزشکی هوشمند هستی.
    پاسخ تو باید دقیقاً مطابق دسته تشخیص‌داده‌شده باشد.

    دسته: {category}

    قوانین پاسخ‌دهی:
    - فقط درباره همین دسته پاسخ بده.
    - از هرگونه توضیح اضافه، پیشنهاد اضافه، جمع‌بندی، یا جملاتی مثل 
      «اگر خواستی»، «می‌خوای انجام بدم؟»، «آیا نیاز به کمک بیشتری داری» پرهیز کن.
    - پاسخ باید کوتاه، دقیق، کاربردی و مستقیم باشد.
    - لحن تخصصی و بدون حاشیه باشد.
    - از دادن توصیه‌های خطرناک پزشکی خودداری کن.
    """

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

