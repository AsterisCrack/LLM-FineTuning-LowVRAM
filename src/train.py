import torch
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from torch.utils.tensorboard import SummaryWriter

try:
    from src.data import get_dataset
except ImportError:
    from data import get_dataset
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

def train(model, tokenized_dataset, tokenizer, device):

    # Enable gradient checkpointing for memory efficiency
    print("Enabling gradient checkpointing...")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Low-Rank Adaptation (LoRA) configuration for efficient fine-tuning
    print("Configuring LoRA...")
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

    # Define DeepSpeed training arguments for the fine-tuning process
    # Using Zero-2 optimization for memory efficiency
    print("Defining DeepSpeed configuration...")
    ds_training_arguments = {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": True
        },
        
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "adam_w_mode": True,
                "lr": 3e-5,
                "betas": [ 0.9, 0.999 ],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": "auto"
            }
        },
        "train_micro_batch_size_per_gpu": "auto",
    }

    # Define training arguments for the fine-tuning process
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir="./model_midtrain_results",  # Directory for saving model checkpoints and logs
        eval_strategy="steps",  # Evaluation strategy: evaluate every few steps
        do_eval=True,  # Enable evaluation during training
        optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
        per_device_train_batch_size=1,  # Batch size per device during training
        gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
        per_device_eval_batch_size=1,  # Batch size per device during evaluation
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        log_level="debug",  # Set logging level to debug for detailed logs
        logging_steps=10,  # Log metrics every 10 steps
        learning_rate=3e-5,  # Initial learning rate
        weight_decay=3e-7,  # Weight decay for regularization
        fp16=True,  # Enable mixed precision training
        eval_steps=100,  # Evaluate the model every 25 steps
        max_steps=100,  # Total number of training steps
        save_steps=25,  # Save checkpoints every 25 steps
        warmup_steps=10,  # Number of warmup steps for learning rate scheduler
        lr_scheduler_type="linear",  # Use a linear learning rate scheduler
        deepspeed=ds_training_arguments,  # DeepSpeed training arguments
        logging_dir='./logs',  # Directory for TensorBoard logs
    )

    # Initialize the Supervised Fine-Tuning (SFT) Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,  # The pre-trained and prepared model
        train_dataset=tokenized_dataset["train"],  # Training dataset
        eval_dataset=tokenized_dataset["test"],  # Evaluation dataset
        peft_config=peft_config,  # LoRA configuration for efficient fine-tuning
        max_seq_length=512,  # Maximum sequence length for inputs
        tokenizer=tokenizer,  # Tokenizer for encoding the data
        args=training_arguments,  # Training arguments defined earlier
    )

    # Start the fine-tuning process
    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        trainer.save_model("./final_interrupted_model")
        print("Model saved.")
        return trainer

    # Save the final model
    print("Saving the model...")
    trainer.save_model("./final_model")

    return trainer

if __name__ == "__main__":
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model name
    model_name = "Qwen/Qwen2.5-7B"
    
    # Load the pre-trained model
    print(f"Loading model {model_name}...")
    # Config for 4 bit quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Config for 8 bit quantization
    nf8_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # Use float16 for mixed precision training
        device_map="auto",  # Distribute the model automatically across GPUs
        # quantization_config=nf4_config,  # Use the bitsandbytes quantization NF4 config
        quantization_config=nf8_config,  # Use the bitsandbytes quantization NF8 config
    )
    print("Model loaded.")

    # Load dataset and tokenizer
    print("Loading dataset and tokenizer...")
    tokenized_dataset, tokenizer = get_dataset(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Dataset and tokenizer loaded.")

    # Train the model
    trainer = train(model, tokenized_dataset, tokenizer, device)

    # TensorBoard logging
    print("Training complete. You can now monitor the training progress in TensorBoard.")
    # To view the TensorBoard logs, run:
    # tensorboard --logdir=./logs
