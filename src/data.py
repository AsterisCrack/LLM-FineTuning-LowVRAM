from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoTokenizer,
)

# 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

def get_dataset(model_name):
    # Load the dataset
    dataset = load_dataset("tatsu-lab/alpaca")

    # Load the tokenizer for Qwen
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,      # Add end-of-sequence token to the tokenizer
        use_fast=True,           # Use the fast tokenizer implementation
        padding_side='left'      # Pad sequences on the left side
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

    def format_conversation(examples):
        # Join the instruction, input and output into a single conversation
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        joined_conversations = [f"{instruction} {input} {output}" for instruction, input, output in zip(instructions, inputs, outputs)]
        # Tokenize the joined conversations
        return tokenizer(joined_conversations, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    # Tokenize the dataset
    tokenized_dataset = dataset["train"].map(format_conversation, batched=True)

    # Remove any columns not needed for training (e.g., original text fields)
    tokenized_dataset = tokenized_dataset.remove_columns(['instruction', 'input', 'output', 'text'])

    # Split intro train and test
    # Dataset is really big, we don't need to lose that much time on evaluation
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.01)
    # print train and test sizes
    print(f"Train size: {len(tokenized_dataset['train'])}")
    print(f"Test size: {len(tokenized_dataset['test'])}")
    
    # Ensure the format is PyTorch-friendly
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return tokenized_dataset, tokenizer
