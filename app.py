from flask import Flask, render_template, request, jsonify, session
from chatbot import Chatbot
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'
chatbot = Chatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    user_message = request.form['user_message']
    bot_response = chatbot.get_response(user_id, user_message)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
