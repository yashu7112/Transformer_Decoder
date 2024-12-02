import itertools
import torch
from torch import nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from datasets import Dataset


def generate_tokens(corpus):
    unique_chars = set(corpus)
    tokens = set()
    for length in range(1, 4): 
        for p in itertools.permutations(unique_chars, length):
            tokens.add("".join(p))
    return list(tokens)


def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def prepare_tokenizer(tokens):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=None, pad_token="<pad>", unk_token="<unk>"
    )
    tokenizer.add_tokens(tokens)
    return tokenizer


def tokenize_corpus(corpus, tokenizer):
    return tokenizer(
        corpus,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def create_model(tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return model


def train_model(model, dataset, epochs=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        for batch in dataset:
            inputs = {
                key: val.unsqueeze(0) for key, val in batch.items()
            }  
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


file_path = "input.txt"  
corpus = load_corpus(file_path)


tokens = generate_tokens(corpus)


tokenizer = prepare_tokenizer(tokens)


tokenized_data = tokenize_corpus(corpus, tokenizer)


model = create_model(tokenizer)


dataset = Dataset.from_dict(
    {
        "input_ids": tokenized_data["input_ids"][0],
        "attention_mask": tokenized_data["attention_mask"][0],
    }
)

train_model(model, dataset, epochs=1)


prompt = "Once upon a time"
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)