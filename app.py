from flask import Flask, render_template, request, session, redirect, url_for

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

app = Flask(__name__)

# Session വേണ്ടി secret key
app.secret_key = "aiml_student_assistant_secret"


faqs = {
    "What is AIML?":
        "AIML stands for Artificial Intelligence and Machine Learning.",

    "What is Machine Learning?":
        "Machine Learning enables computers to learn from data.",

    "What is NLP?":
        "NLP stands for Natural Language Processing.",

    "What is Data Science?":
        "Data Science is the study of extracting insights from data.",

    "What is Deep Learning?":
        "Deep Learning is a subset of Machine Learning.",

    "What is TensorFlow?":
        "TensorFlow is a popular machine learning framework developed by Google.",

    "What is PyTorch?":
        "PyTorch is an open-source deep learning framework.",

    "What is Computer Vision?":
        "Computer Vision enables computers to understand images and videos.",

    "What are career opportunities in AIML?":
        "AI Engineer, ML Engineer, Data Scientist and NLP Engineer.",

    "What projects can AIML students do?":
        "Chatbots, Recommendation Systems, Image Classifiers and Sentiment Analysis."
}


@app.route("/", methods=["GET", "POST"])
def home():

    # Session-il chat history illenkil create cheyyuka
    if "chats" not in session:
        session["chats"] = []

    if request.method == "POST":

        question = request.form["question"]

        tokens = word_tokenize(question.lower())
        processed_question = " ".join(tokens)

        questions = list(faqs.keys())

        vectorizer = TfidfVectorizer()

        faq_vectors = vectorizer.fit_transform(questions)

        user_vector = vectorizer.transform([processed_question])

        similarity = cosine_similarity(
            user_vector,
            faq_vectors
        )

        score = similarity.max()

        index = similarity.argmax()

        if score > 0.25:
            answer = faqs[questions[index]]
        else:
            answer = "Sorry, I don't have information about that topic."

        chats = session["chats"]

        chats.append({
            "question": question,
            "answer": answer
        })

        session["chats"] = chats
        session.modified = True

    return render_template(
        "index.html",
        chats=session["chats"]
    )


@app.route("/new_chat")
def new_chat():
    session["chats"] = []
    session.modified = True
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)