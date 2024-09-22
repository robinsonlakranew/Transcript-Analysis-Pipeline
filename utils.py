'''
This file contains the preprocess_transcript function, which is 
responsible for cleaning and extracting relevant portions from 
the transcript JSON.

The classify_objection function is responsible for 
classifying the objections into different categories.

extract_customer_requirements module will handle the customer 
requirement extraction using the LLaMA model.

determine_conversation_outcome is for determining the conversation 
outcome based on the final customer statements
'''

import json

def preprocess_review(review_json):
    """
    Extract relevant text from the review JSON.
    """
    conversation = []
    for entry in review_json:
        start = entry['start']
        end = entry['end']
        text = entry['text'].strip()
        conversation.append({'start': start, 'end': end, 'text': text})
    return conversation

def classify_objection(text, model, tokenizer):
    """
    Classifies objections into categories: 'Car Not Available', 
                                            'Price Dissatisfaction', 
                                            'Service Issue'
    """
    prompt = f"Categorize the objection in this text: '{text}' as one of the following categories: Car Not Available, Price Dissatisfaction, Service Issue."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Car Not Available" in generated_text:
        return "Car Not Available"
    elif "Price Dissatisfaction" in generated_text:
        return "Price Dissatisfaction"
    elif "Service Issue" in generated_text:
        return "Service Issue"
    else:
        return "Unknown"

def extract_customer_requirements(text, model, tokenizer):
    """
    Extract customer requirements from the transcript.
    """
    prompt = f"Extract the customer requirements for a car in the following text: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    
    requirements = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return requirements


def determine_conversation_outcome(text, model, tokenizer):
    """
    Determine the outcome of the conversation (e.g., purchase intent, undecided, visiting other dealerships).
    """
    prompt = f"What is the outcome of this conversation? The options are: Customer intends to make a purchase, Customer will visit other dealerships, Customer needs more time to decide. Conversation {text}"
    
    # combined_text = " ".join([entry['text'] for entry in conversation[-3:]])
    # prompt += combined_text
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    
    outcome = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return outcome
