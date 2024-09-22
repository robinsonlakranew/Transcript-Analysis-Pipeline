'''
The main.py file is the entry point where everything comes together. 
Here we load the transcript data, call the relevant functions 
from other files, and display the results.
'''

import json, os
from models.llama_model import load_llama_model
from utils import preprocess_review, \
                    classify_objection, \
                    extract_customer_requirements, \
                    determine_conversation_outcome


# Function to analyze a transcript entry by entry and return the structured output
def analyze_transcript(transcript):
    """
    Analyze a transcript entry-by-entry and extract objections, customer requirements, and conversation outcome.
    """
    # Load LLaMA model and tokenizer
    model, tokenizer = load_llama_model()

    # Preprocess the transcript
    conversation = preprocess_transcript(transcript)

    # Store results for each entry
    results = []

    # Process each entry in the conversation individually
    for entry in conversation:
        # 1. Classify objection
        objection_category = classify_objection(entry['text'], model, tokenizer)

        # 2. Extract customer requirements
        customer_requirements = extract_customer_requirements(entry['text'], model, tokenizer)

        # 3. Determine conversation outcome
        conversation_outcome = determine_conversation_outcome(entry['text'], model, tokenizer)

        # Store the result for the current entry
        result = {
            "entry": {
                "start_time": entry['start'],
                "end_time": entry['end'],
                "text": entry['text']
            },
            "objection_category": objection_category,
            "customer_requirements": customer_requirements,
            "conversation_outcome": conversation_outcome
        }
        results.append(result)

    return results

# Function to handle multiple files from the 'data' folder
def process_transcripts(input_folder, output_folder):
    """
    Process all transcripts in the input folder and save the output in the output folder.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each JSON file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_file_path = os.path.join(input_folder, filename)
            
            # Read the JSON transcript
            with open(input_file_path, 'r') as f:
                transcript_data = json.load(f)

            # Analyze the transcript
            analysis = analyze_transcript(transcript_data)

            # Define the output filename (add '_output' before '.json')
            output_filename = f"{filename.split('.json')[0]}_output.json"
            output_file_path = os.path.join(output_folder, output_filename)

            # Write the structured analysis to the output folder
            with open(output_file_path, 'w') as f:
                json.dump(analysis, f, indent=4)

            print(f"Processed {filename} -> {output_filename}")

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = 'data'
    output_folder = 'output'

    # Process all transcripts in the input folder
    process_transcripts(input_folder, output_folder)
