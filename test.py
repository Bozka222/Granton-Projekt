from openai import OpenAI
from flask import Flask, request, render_template
import os

open_AI_key = os.environ.get("OPENAI")
client = OpenAI(api_key=open_AI_key)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/ask', methods=['POST'])
def ai_response():
    data = request.get_json()
    message = data['message']

    def generate():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    return generate()  # Calls generate API function


if __name__ == '__main__':
    app.run(debug=True)
