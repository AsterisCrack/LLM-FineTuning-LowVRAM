from transformers import AutoModelForCausalLM
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from peft import LoraConfig, get_peft_model
import torch

# Define your LoRA config
peft_config = LoraConfig(
    lora_alpha=32,  # Scaling factor for LoRA updates
    lora_dropout=0.05,  # Dropout rate applied to LoRA layers
    r=64,  # Rank of the LoRA decomposition
    bias="none",  # No bias is added to the LoRA layers
    task_type="CAUSAL_LM",  # Specify the task as causal language modeling
    target_modules=[  # Modules to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print("Loading model...")
model_name = "Qwen/Qwen2.5-3B"
    
# Load the pre-trained model
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set this to True to load model in 8-bit
    bnb_4bit_use_double_quant=False,  # Enable double quantization if you need 4-bit precision (optional)
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use float16 for mixed precision training
    device_map="auto",  # Distribute the model automatically across GPUs
)
print("Model loaded.")

# Apply LoRA to the model
print("Applying LoRA configuration...")
model = get_peft_model(model, peft_config)
print("LoRA applied.")

# Optionally, you can print the modified model to inspect it
# print(model)

# Estimating memory requirements based on LoRA's updates
print("Estimating memory requirements for LoRA...")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
