import os
import json
import math
import torch
import numpy as np
import argparse
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass

# Configuration class
@dataclass
class ModelConfig:
    n_dataset: int = 1000000
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 512
    max_seq_length: int = 64
    epochs: int = 50
    seed: int = 42
    dataset_path: str = ""  # Will be generated dynamically
    output_dir: str = ""  # Will be generated dynamically
    tokenizer_dir: str = f'tokenizer_language2motion_n{n_dataset}'

    def __post_init__(self):
        # Generate output directory if not provided
        if not self.output_dir:
            self.output_dir = (f'model_object_extraction_n{self.n_dataset}_d{self.d_model}_h{self.nhead}_'
                             f'l{self.num_encoder_layers}_f{self.dim_feedforward}_msl{self.max_seq_length}_hold_success')
        if not self.dataset_path:
            self.dataset_path = f'generated_vlm_dataset_n{self.n_dataset}_cxn025/vison_language_motion_pair_format_split82_n{self.n_dataset}'

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as non-trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]

# Transformer model
class SequenceToSequenceClassTransformer(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Classification head
        self.class_head = nn.Linear(config.d_model, num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        # Input embedding + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        # Use [CLS] token representation (first token)
        x = x[:, 0]
        
        # Generate class logits
        logits = self.class_head(x)
        return logits

# Dataset class
class VLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, class_mapping):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use mission instruction as input
        input_str = item['mission_instruction_1']
        
        # Tokenize input
        encoding = self.tokenizer(
            input_str,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get class ID for mission object
        mission_object_1 = item['mission_object_1']
        class_id = self.class_mapping[mission_object_1]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(class_id, dtype=torch.long)
        }

# Training function
def train_model(config):
    # Set random seeds for reproducibility
    print('Setting random seeds...')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    print('Creating output directory...')
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration
    print('Saving configuration...')
    with open(os.path.join(config.output_dir, 'config_object_extraction_transformer.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)

    # Load dataset
    print('Loading dataset...')
    # dataset_path = f'generated_vlm_dataset_n{config.n_dataset}/vison_language_motion_pair_format_split82_n{config.n_dataset}'
    dataset = load_from_disk(config.dataset_path)

    # Load tokenizer
    print('Loading tokenizer...')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

    # Build class mapping from mission objects
    mission_objects = set([item['mission_object_1'] for item in dataset['train']])
    class_mapping = {obj: i for i, obj in enumerate(mission_objects)}
    num_classes = len(class_mapping)
    
    print(f'Class mapping: {class_mapping}')
    print(f'Number of classes: {num_classes}')
    
    # Save class mapping
    with open(os.path.join(config.output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)

    # Create datasets
    print('Creating datasets...')
    train_dataset = VLMDataset(dataset['train'], tokenizer, config.max_seq_length, class_mapping)
    test_dataset = VLMDataset(dataset['test'], tokenizer, config.max_seq_length, class_mapping)

    # Create data loaders
    print('Creating data loaders...')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Initialize model
    print('Initializing model...')
    model = SequenceToSequenceClassTransformer(config, tokenizer.vocab_size, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    # Optimizer and loss function
    print('Setting up optimizer and loss function...')
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print('Starting training loop...')
    best_test_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            # Move data to device
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs, masks)
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluation phase
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                logits = model(inputs, masks)
                test_loss += criterion(logits, labels).item()

        # Calculate average losses
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)

        # Save best model
        if avg_test <= best_test_loss:
            best_test_loss = avg_test
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'object_extraction_transformer.pth'))
            print(f'Saved best model at Epoch {epoch + 1}')

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{config.epochs}:\n'
              f'Train Loss: {avg_train:.4f}\n'
              f'Test  Loss: {avg_test:.4f}\n'
              f'----------------------------------------')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Object Extraction Transformer')
    
    # Add arguments for all configuration parameters
    parser.add_argument('--n_dataset', type=int, default=1000000, help='Dataset size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_path', type=str, default="", help='Dataset path')
    parser.add_argument('--output_dir', type=str, default="", help='Output directory')
    parser.add_argument('--tokenizer_dir', type=str, default="", help='Tokenizer directory')
    
    args = parser.parse_args()
    
    # Handle default tokenizer directory
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else f'tokenizer_language2motion_n{args.n_dataset}'
    
    # Create configuration from arguments
    config = ModelConfig(
        n_dataset=args.n_dataset,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        tokenizer_dir=tokenizer_dir
    )
    
    # Start training
    train_model(config)

if __name__ == '__main__':
    main()