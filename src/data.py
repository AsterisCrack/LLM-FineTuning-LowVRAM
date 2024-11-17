from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoTokenizer,
)

def get_dataset(model_name):
    # Load the dataset
    dataset = load_dataset("GAIR/lima")

    # Load the tokenizer for Qwen
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,      # Add end-of-sequence token to the tokenizer
        use_fast=True,           # Use the fast tokenizer implementation
        padding_side='left'      # Pad sequences on the left side
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

    def format_conversation(examples):
        # Join the list into a single string if it's a list of sentences
        joined_conversations = [" ".join(conv) if isinstance(conv, list) else conv for conv in examples['conversations']]
        
        # Tokenize the joined conversations
        return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    # Tokenize the dataset
    tokenized_dataset = dataset.map(format_conversation, batched=True)

    # Remove any columns not needed for training (e.g., original text fields)
    tokenized_dataset = tokenized_dataset.remove_columns(["conversations", "source"])

    # Ensure the format is PyTorch-friendly
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return tokenized_dataset, tokenizer
