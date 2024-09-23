'''
This file will handle loading the LLaMA model and tokenizer.

'''
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model(device):
    """
    Load the LLaMA model and tokenizer from Hugging Face.
    """

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    
    return model, tokenizer                                                                                 "I had booked it for 3.5 to 5 lakhs. I had shortlisted Jazz. I had shortlisted... What is the rate of Aspire? 4.5, 5.5. Jazz is better. Because HOT's company is also closing down. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case deck. I had booked a white Jazz case", 'conversation_outcome': "Categorize the objection in this text: 'I had booked it for 3.5 to 5 lakhs. I had shortlisted Jazz. I had shortlisted... What is the rate of Aspire? 4.5, 5.5. Jazz is better. Because HOT's company is also closing down. I had booked a white Jazz case deck.' as one of the following categories: Customer intends to make a purchase, Customer will visit other dealerships, Customer needs more time to decide.\n\nThe customer is not aware of the price of the product.\n\nThe customer is not aware of the price of the product.\n"}





