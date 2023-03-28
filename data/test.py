def process_txt(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            modified_line = line.replace("Question:", "@Question:").replace("Response:", "@").replace("Label:","@")
            outfile.write(modified_line)

# Example usage:

process_txt("data/prompt_answer_150.csv", "data/prompt_answer_150_II.csv")
