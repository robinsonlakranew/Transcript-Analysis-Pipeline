'''
This file will handle loading the LLaMA model and tokenizer.

'''
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model():
    """
    Load the LLaMA model and tokenizer from Hugging Face.
    """

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    
    return model, tokenizer
