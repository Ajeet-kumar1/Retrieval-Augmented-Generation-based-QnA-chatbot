import json
import requests
from utils import url




def inference(url, prompt):
    payload = {
        "model": 'llama1',
        #"model": 'deepseek-r1:1.5b',
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


if __name__=='__main__':
    # Define the request parameters
    notQuit = True
    while notQuit==True:
        prompt = input("You>> ")
        if prompt=="exit":
            notQuit = False
        else:
            final_response = inference(url, prompt)
            print(f"Bot>> {final_response}")