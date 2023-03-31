from datasets import load_dataset 

# Load the full dataset
full_dataset = load_dataset('csv', data_files='data/prompt_answer_150.csv', delimiter='@', column_names=[
                            "movie_name", "question", "answer"], cache_dir="./cache",split='train')

# Split the dataset into train and validation sets
full_dataset = full_dataset.train_test_split(test_size=0.2, shuffle=True)
full_dataset

