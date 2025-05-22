import os
import ollama
import threading
import webbrowser
from pdf_process import split_documents, load_pdf_documents
from faiss_process import create_vector_store, retrieve_context
from flask import Flask, request, render_template, session, redirect, url_for


pdf_path = input("Enter the pdf path: ")


#pdf_path = os.getenv("PDF_PATH", "default.pdf")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session


# Generate answer using Ollama (unchanged)
def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context)
    
    prompt = f"""You are an expert assistant trained on document information.
    Use this context to answer the question:
    
    {formatted_context}
    
    Question: {query}
    
    Answer in detail using only the provided context:"""
    
    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt= prompt,
        options={
            'temperature': 0.3,
            'max_tokens': 2000
        }
    )
    return response['response']



# Main function
def main(pdf_path, query):
    # Load and process PDF
    pages = load_pdf_documents(pdf_path)  # Get Document objects
    split_docs = split_documents(pages)   # Split properly
    
    # Create vector store
    index, document_texts, embedder = create_vector_store(split_docs)
    
    # Retrieve context
    context = retrieve_context(query, embedder, index, document_texts)
    
    # Generate answer
    answer = generate_answer_with_ollama(query, context)
    return answer


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_input = request.form['message']
        bot_response = main(pdf_path, user_input)

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