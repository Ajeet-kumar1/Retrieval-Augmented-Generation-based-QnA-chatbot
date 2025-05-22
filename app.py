import os
import ollama
import threading
import webbrowser
from pdf_process import split_documents, load_pdf_documents
from faiss_process import create_vector_store, retrieve_context
from flask import Flask, request, render_template, session, redirect, url_for


pdf_path = input("Enter the pdf path: ")
#pdf_path = r"C:/Users/Ajeet/Downloads/attention-is-all-you-need.pdf"

#pdf_path = os.getenv("PDF_PATH", "default.pdf")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session

# Answer generation with
def generate_answer_with_ollama(query, context, chat_history):
    formatted_context = "\n".join(context)
    
    # Format previous turns into a running history
    history = ""
    for speaker, message in chat_history[-6:]:  # Only last 3 turns (You, Bot, You...) to keep prompt short
        history += f"{speaker}: {message}\n"

    prompt = f"""You are an expert assistant trained on document information.
Use the following document context and prior conversation to answer the user's question.

Document Context:
{formatted_context}

Chat History:
{history}
You: {query}
Bot:"""

    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={
            'temperature': 0.3,
            'max_tokens': 2000
        }
    )
    return response['response']


# main function
def main(pdf_path, query, chat_history):
    # Load and process PDF
    pages = load_pdf_documents(pdf_path)
    split_docs = split_documents(pages)
    index, document_texts, embedder = create_vector_store(split_docs)

    # Retrieve context
    context = retrieve_context(query, embedder, index, document_texts)

    # Generate answer using chat memory
    answer = generate_answer_with_ollama(query, context, chat_history)
    return answer


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_input = request.form['message']
        chat_history = session['chat_history']
        bot_response = main(pdf_path, user_input, chat_history)

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