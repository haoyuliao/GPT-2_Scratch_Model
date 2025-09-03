# -*- coding: utf-8 -*-

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023–2025 Sebastian Raschka
# Modifications (2025-09-02) by Howard Liao: refactor/reorg.

import yaml
import tiktoken
import torch
import torch.nn as nn
from models import GPTModel
from scripts import generate_text_simple
from scripts import text_to_token_ids
from scripts import token_ids_to_text
from scripts import GPTDatasetV1
from scripts import create_dataloader_v1
from scripts import calc_loss_loader

def main():
    with open("config/gpt2_124M.yaml", "r") as f:
        GPT_CONFIG_124M = yaml.safe_load(f)

    file_path = "./dataset/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)
    
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.
    
    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    #
    # print(f"Using {device} device.")
    
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
    
    
    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader
    
    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
        
    """
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    #start_context = "Hello, I am"
    start_context = "Every effort moves you"
    
    
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
    
    

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    """


if __name__ == "__main__":
    main()