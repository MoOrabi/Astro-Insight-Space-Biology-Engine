from flask import Flask, request
from flask_cors import CORS

import test_query
app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['GET'])
def get_question_response():
    question = request.args.get('question')  # هنا ناخدها من query string
    return test_query.get_question_answer(question)


@app.route('/ask-test', methods=['GET'])
def get_question_response_test():
    data = request.get_json()
    citations = []
    citations_names_with_year = []

    metas = [
        {
            "id": "doc1",
            "url": "https://nasa.gov/pub123",
            "title": "Bone Density in Space",
            "author": "Smith et al.",
            "year": "2018"
        },
        {
            "id": "doc2",
            "url": "https://nasa.gov/pub456",
            "title": "Countermeasures for Bone Loss",
            "author": "Johnson & Lee",
            "year": "2020"
        }
    ]
    for m in metas:
        citations.append({
            "url": m.get("url", ""),
            "title": m.get("title", "")
        })
        citations_names_with_year.append({
            "title": m.get("title", ""),
            "year": m.get("year", "")
        })

    return {
        "answer": "Your connection to backend is working alright, and your question is " + data.get("question")
        + ", Good Job",
        "citations": citations,
        "citationsNamesWithYear": citations_names_with_year
    }


app.run(debug=True)