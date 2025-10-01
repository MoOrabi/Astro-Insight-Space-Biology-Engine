from flask import Flask, request
from flask_cors import CORS

import test_query
app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['GET'])
def get_question_response():
    data = request.get_json()
    question = data.get('question')
    return test_query.get_question_answer(question)


app.run(debug=True)