from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig
import json
from tqdm import tqdm

# Step 1: Load the tokenizer and model with quantization
# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Model name
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the pre-trained model
print(f"Loading model {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set this to True to load model in 8-bit
    bnb_4bit_use_double_quant=False,  # Enable double quantization if you need 4-bit precision (optional)
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use float16 for mixed precision training
    device_map=device,  # Distribute the model automatically across GPUs
    # quantization_config=bnb_config,  # Use the bitsandbytes quantization config
)
model.config.pad_token_id = tokenizer.pad_token_id
print("Model loaded.")

# Step 2: Get trained LORA and BNB model
lora_location = "final_model/"
model.load_adapter(lora_location)
print("LoRA and BNB model loaded.")

# Step 3: Load the google/IFEval dataset
dataset = load_dataset("google/IFEval")

# Step 4: Generate predictions on the dataset
output_file = "model_responses.jsonl"
with open(output_file, 'w', encoding='utf-8') as f_out:
    for sample in tqdm(dataset['train']):   # Use 'validation' or 'train' split if 'test' is not available
        input_text = sample['prompt']  # Adjust the field name based on the dataset's structure

        # Prepare the input prompt
        prompt = input_text

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        attention_mask = inputs["attention_mask"]
        # Generate output
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=256,
        )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Since the model may include the prompt in its output, we extract the generated response
        response = generated_text[len(prompt):]

        # Prepare the JSON object
        json_obj = {
            "prompt": prompt,
            "response": response
        }

        # Write the JSON object to file
        f_out.write(json.dumps(json_obj) + '\n')