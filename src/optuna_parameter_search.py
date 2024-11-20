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
import optuna
try:
    from src.data import get_dataset
except ImportError:
    from data import get_dataset

def objective(trial, model, tokenized_dataset, tokenizer, device):

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

    # Define training arguments for the fine-tuning process
    # Find the best hyperparameters using Optuna
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir="./model_midtrain_results",  # Directory for saving model checkpoints and logs
        eval_strategy="steps",  # Evaluation strategy: evaluate every few steps
        do_eval=True,  # Enable evaluation during training
        optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
        per_device_train_batch_size=2,  # Batch size per device during training
        gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
        per_device_eval_batch_size=2,  # Batch size per device during evaluation
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        log_level="debug",  # Set logging level to debug for detailed logs
        logging_steps=10,  # Log metrics every 10 steps
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3),  # Initial learning rate
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),  # Weight decay for regularization
        eval_steps=101,  # Evaluate the model every 25 steps
        max_steps=100,  # Total number of training steps
        save_steps=101,  # Save checkpoints every 25 steps
        warmup_steps=5,  # Number of warmup steps for learning rate scheduler
        lr_scheduler_type="linear",  # Use a linear learning rate scheduler
        logging_dir='./logs',  # Directory for TensorBoard logs
    )
        
    # Initialize the Supervised Fine-Tuning (SFT) Trainer
    print("Initializing trainer...")
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
    print("Starting training...")
    trainer.train()
    return trainer.evaluate().get("eval_loss")
    
            
    

if __name__ == "__main__":
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model name
    model_name = "Qwen/Qwen2.5-3B"
    
    # Load the pre-trained model
    print(f"Loading model {model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Set this to True to load model in 8-bit
        bnb_4bit_use_double_quant=False,  # Enable double quantization if you need 4-bit precision (optional)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # Use float16 for mixed precision training
        device_map="auto",  # Distribute the model automatically across GPUs
        # quantization_config=bnb_config,  # Use the bitsandbytes quantization config
    )
    print("Model loaded.")

    # Load dataset and tokenizer
    print("Loading dataset and tokenizer...")
    tokenized_dataset, tokenizer = get_dataset(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Dataset and tokenizer loaded.")

    # Look for the best hyperparameters using Optuna
    print("Starting hyperparameter search...")
    study = optuna.create_study(study_name="hyperparameter_search", direction="minimize")
    study.optimize(lambda trial: objective(trial, model, tokenized_dataset, tokenizer, device), n_trials=10)

    print("Best hyperparameters found:")
    print(study.best_value)
    print(study.best_params)
    print(study.best_trial)
    """Best hyperparameters found:
2.5227315425872803
{'learning_rate': 0.0009137026293794083, 'weight_decay': 0.030224260406242976}
FrozenTrial(number=0, state=1, values=[2.5227315425872803], datetime_start=datetime.datetime(2024, 11, 18, 19, 21, 27, 135313), datetime_complete=datetime.datetime(2024, 11, 18, 19, 25, 55, 841593), params={'learning_rate': 0.0009137026293794083, 'weight_decay': 0.030224260406242976}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.001, log=False, low=1e-05, step=None), 'weight_decay': FloatDistribution(high=0.1, log=False, low=0.0, step=None)}, trial_id=0, value=None)
"""
"""
Best hyperparameters found:
2.4710028171539307
{'learning_rate': 0.0007351748905402193, 'weight_decay': 0.0817835012385977}
FrozenTrial(number=8, state=1, values=[2.4710028171539307], datetime_start=datetime.datetime(2024, 11, 18, 21, 14, 53, 535754), datetime_complete=datetime.datetime(2024, 11, 18, 21, 22, 29, 625064), params={'learning_rate': 0.0007351748905402193, 'weight_decay': 0.0817835012385977}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'learning_rate': FloatDistribution(high=0.001, log=False, low=1e-05, step=None), 'weight_decay': FloatDistribution(high=0.1, log=False, low=0.0, step=None)}, trial_id=8, value=None)"""
