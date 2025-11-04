from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(api_key="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")

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'bac.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'پیام خالی است'}), 400

    response = client.responses.create(
        model="gpt-5-nano",
        input=user_message
    )

    return jsonify({'reply': response.output_text.strip()})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
