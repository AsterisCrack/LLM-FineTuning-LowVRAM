from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig
import json
from tqdm import tqdm
import time

def get_response(model, tokenizer, device, prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    attention_mask = inputs["attention_mask"]
    # Generate output
    t0 = time.time()
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.7,
        repetition_penalty=1.25,
        streamer=streamer,
    )

    return streamer, time.time() - t0

if __name__ == "__main__":
    # Step 1: Load the tokenizer and model with quantization
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model name
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

    # Load the pre-trained model
    print(f"Loading model {model_name}...")
    # Config for 8 bit quantization
    nf8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    # Step 2: Get trained LORA and BNB model
    lora_location = "final_model/"
    model = AutoModelForCausalLM.from_pretrained(
        lora_location, 
        torch_dtype=torch.float16,  # Use float16 for mixed precision training
        device_map=device,  # Distribute the model automatically across GPUs
        quantization_config=nf8_config,  # Use the bitsandbytes quantization config
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")

    print("LoRA and BNB model loaded.")

    # Obtain the prompt from the user
    while True:
        prompt = input("Enter your prompt: ")
        if prompt == "exit":
            break
        
        try:
            response, response_time = get_response(model, tokenizer, device, prompt)
            print(f"Response time: {response_time:.2f} seconds")
        except KeyboardInterrupt:
            print("Interrupted.")