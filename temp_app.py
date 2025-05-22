from flask import Flask, request, render_template, session, redirect, url_for
import threading
import webbrowser
import json
import requests
from utils import url


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session


# Inference
def inference(url, prompt):
    payload = {
        #"model": 'llama1',
        "model": 'deepseek-r1:1.5b',
        "prompt": prompt,
        "temperature": 0.5,
        "max_tokens": 10
    
    }
    # Send the request
    response = requests.post(url, json=payload)
    final_response = ""
    for chunk in response.text.split("\n"):
        if chunk.strip():  # Skip empty lines
            data = json.loads(chunk)
            final_response += data.get("response", "")
    return final_response


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_input = request.form['message']
        bot_response = inference(url, user_input)

        session['chat_history'].append(("You", user_input))
        session['chat_history'].append(("Bot", bot_response))
        session.modified = True

        return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=session['chat_history'])

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=False)
