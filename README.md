# Retrieval-Augmented-Generation-based-QnA-chatbot

This is a Retrieval-Augmented-Generation based QnA chatbot. This meant for logical and mathematical question answering. This chatbot is very helpful when logical thinking is required.

In this chatbot as LLM deepseek and faiss as used vector database is used. 

To run this follow these command
if you want to run in conda environment then follow these commands.
Install ollama from here https://ollama.com/download
```
git clone https://github.com/Ajeet-kumar1/Retrieval-Augmented-Generation-based-QnA-chatbot.git
cd Retrieval-Augmented-Generation-based-QnA-chatbot
```


```
conda create -n rag python==3.10
conda activate rag
pip install -r requirements.txt
```

```
python app.py
```