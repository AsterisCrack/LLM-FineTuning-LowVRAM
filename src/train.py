import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from src.data import get_dataset


def train(model, tokenized_dataset, tokenizer, device):
    # Send things to device
    model.to(device)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Low-Rank Adaptation (LoRA) configuration for efficient fine-tuning
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
    #We are using Zero-2 optimization for memory efficiency
    ds_training_arguments = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "round_robin_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }
    
    # Define training arguments for the fine-tuning process
    training_arguments = TrainingArguments(
        output_dir="./qwen7B_results",  # Directory for saving model checkpoints and logs
        eval_strategy="steps",  # Evaluation strategy: evaluate every few steps
        do_eval=True,  # Enable evaluation during training
        optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
        per_device_train_batch_size=4,  # Batch size per device during training
        gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
        per_device_eval_batch_size=2,  # Batch size per device during evaluation
        log_level="debug",  # Set logging level to debug for detailed logs
        logging_steps=10,  # Log metrics every 10 steps
        learning_rate=1e-4,  # Initial learning rate
        eval_steps=25,  # Evaluate the model every 25 steps
        max_steps=100,  # Total number of training steps
        save_steps=25,  # Save checkpoints every 25 steps
        warmup_steps=25,  # Number of warmup steps for learning rate scheduler
        lr_scheduler_type="linear",  # Use a linear learning rate scheduler
        deepspeed=ds_training_arguments,  # DeepSpeed training arguments
    )
    # Initialize the Supervised Fine-Tuning (SFT) Trainer
    trainer = SFTTrainer(
        model=model,  # The pre-trained and prepared model
        train_dataset=tokenized_dataset["train"],  # Training dataset
        eval_dataset=tokenized_dataset["test"],  # Evaluation dataset
        peft_config=peft_config,  # LoRA configuration for efficient fine-tuning
        max_seq_length=2048,  # Maximum sequence length for inputs
        tokenizer=tokenizer,  # Tokenizer for encoding the data
        args=training_arguments,  # Training arguments defined earlier
    )

    # Start the fine-tuning process
    trainer.train()

    # Save the final model
    trainer.save_model("./qwen7B_final_model")

    return trainer


if __name__ == "__main__":

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model name
    model_name = "Qwen/Qwen2.5-7B"
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"  # Automatically map layers to devices
    )

    tokenized_dataset, tokenizer = get_dataset(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = train(model, tokenized_dataset, tokenizer, device)
