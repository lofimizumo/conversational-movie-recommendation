import torch
from model.responseModel import MovieResponseModel,MovieResponseConfig
from transformers import AutoTokenizer


def generate_response(model, tokenizer, input_question, max_length=50):

    # Tokenize the input question
    input_tokens = tokenizer.encode(input_question, return_tensors="pt")
    
    # Generate the response
    with torch.no_grad():
        output_tokens = model.generate(input_tokens, max_length=max_length)
    
    # Decode the generated tokens back to text
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return response

# Load the checkpoint and create a model instance

checkpoint_path = "lightning_logs/version_2/checkpoints/best-checkpoint.ckpt"
model_name = "google/flan-t5-small"


model = MovieResponseModel.load_from_checkpoint(checkpoint_path)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the input question
input_question = "Can you recommend a movie with a mind-bending plot and great acting?"

# Generate the response
response = generate_response(model, tokenizer, input_question)
print(response)
