from collections import OrderedDict
from util.dataset import read_plots_from_csv
from model.responseConfig import MovieResponseConfig
from model.responseModel import MovieResponseModel
from transformers import AutoTokenizer
import torch
import pandas as pd

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*",
     "methods": ["POST"], "allow_headers": ["Content-Type"]}})

# Load the model during API initialization


def generate_response(model, tokenizer, input_question, max_length=50):

    # Tokenize the input question
    input_tokens = tokenizer.encode(
        input_question, return_tensors="pt", max_length=2048, truncation=True)

    # Generate the response
    with torch.no_grad():
        output_tokens = model.model.generate(
            input_tokens, max_length=max_length)

    # Decode the generated tokens back to text
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return response


@app.before_first_request
def load_model():
    # Load the checkpoint and create a model instance
    # Load the checkpoint and create a model instance
    checkpoint_path = "responseModel.ckpt"
    model_name = "google/flan-t5-small"
    print("ResponseModel loaded")

    global model, tokenizer
    model = torch.load(checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint_path = "best.ckpt"
    global rec_model
    rec_model = torch.load(checkpoint_path)
    rec_model.eval()
    print("RecModel loaded")


@app.route('/')
def index():
    return 'Index Page'


@app.route('/predict', methods=['POST','OPTIONS'])
def predict():

    if request.method == 'OPTIONS':
        # Handle pre-flight request
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    def load_movies(csv_file):
        return pd.read_csv(csv_file)

    def find_movie_description(movies, movie_name):
        match = movies[movies['title'].str.contains(movie_name, case=False)]
        if not match.empty:
            return match.iloc[0]['plot_synopsis']
        else:
            return None

    def make_question(question, movie_name, movies):
        movie_description = find_movie_description(movies, movie_name)
        if movie_description:
            return f"{question} Recommended Movie: {movie_name}. Movie Description: {movie_description}"
        else:
            return f"{question} Sorry, we couldn't find a movie with that name."

    def pred(question, model):
        movie_dict = read_plots_from_csv('data/movie_1000_clean.csv')
        names = movie_dict.keys()
        idx = list(range(len(names)))
        idx_name_dict = OrderedDict(zip(idx, names))
        ret = model.inference(question)
        ret = ret.reshape(-1)
        ret = list(ret)
        ret = [int(x) for x in ret]
        pred_names = [idx_name_dict[int(x)] for x in ret]
        return pred_names[0]

    # input_question = "Can you recommend some scary movie about a murderer killing everyone"
    input_question = request.json['input_question']
    movie_name = pred(input_question, rec_model)
    print(movie_name)

    movie_database = load_movies('data/mpst_full_data.csv')
    input_question_with_prompt = make_question(
        input_question, movie_name, movie_database)

    response = generate_response(model, tokenizer, input_question_with_prompt)

    response_json = {"answer": response,"movie_name":movie_name}
    return jsonify(response_json)
