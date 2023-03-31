import requests

input_question = "Can you recommend some scary movie about a murderer killing everyone"
url = 'http://127.0.0.1:5000/predict'
response = requests.post(url, json={'input_question': input_question})
print(response.json())
