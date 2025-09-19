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
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from dataclasses import dataclass

# Configuration class
@dataclass
class ModelConfig:
    n_dataset: int = 1000000
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 512
    max_seq_length: int = 64
    epochs: int = 50
    seed: int = 42
    output_dir: str = ""  # Will be generated dynamically
    tokenizer_dir: str = ""  # Will be generated dynamically
    beta: int = 10  # Weight for motion loss

    def __post_init__(self):
        # Generate directories if not provided
        if not self.output_dir:
            self.output_dir = (f'model_language2motion_n{self.n_dataset}_d{self.d_model}_h{self.nhead}_'
                              f'l{self.num_layers}_f{self.dim_feedforward}_msl{self.max_seq_length}_'
                              f'hold_success_cxn025_beta{self.beta}')
        if not self.tokenizer_dir:
            self.tokenizer_dir = f'tokenizer_language2motion_n{self.n_dataset}'

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
class LanguageToMotionTransformer(nn.Module):
    def __init__(self, config, vocab_size):
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
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Prediction heads
        self.motion_head = nn.Linear(config.d_model, 3)  # 3D motion vector
        self.mission_state_head = nn.Linear(config.d_model, 4)  # 4 possible mission states
        self.search_state_head = nn.Linear(config.d_model, 2)  # 2 possible search states
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        # Input embedding + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        # Use [CLS] token representation (first token)
        x = x[:, 0]
        
        # Generate predictions
        motion = self.motion_head(x)
        mission_state = self.mission_state_head(x)
        search_state = self.search_state_head(x)
        
        return motion, mission_state, search_state

# Dataset class
class VLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct input string with all relevant features
        input_str = (
            f"{item['mission_instruction_0']} [SEP] {item['mission_instruction_1']} "
            f"[SEP] {item['predicted_object']} [SEP] confidence:{item['confidence'][0]:.2f} "
            f"[SEP] object_xyn:{item['object_xyn'][0]:.2f} {item['object_xyn'][1]:.2f} "
            f"[SEP] object_whn:{item['object_whn'][0]:.2f} {item['object_whn'][1]:.2f} "
            f"[SEP] {item['mission_state_in']}"
            f"[SEP] {item['search_state_in']}"
        )
        
        # Tokenize input
        encoding = self.tokenizer(
            input_str,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process outputs
        motion = torch.tensor(item['motion_vector'], dtype=torch.float)
        
        # Encode mission states as integers
        mission_state = 0 if item['mission_state_out'] == 'success' else \
                        1 if item['mission_state_out'] == 'searching_1' else \
                        2 if item['mission_state_out'] == 'searching_0' else 3
                        
        # Encode search states as integers
        search_state = 0 if item['search_state_out'] == 'had_searching_1' else 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'motion': motion,
            'mission_state': torch.tensor(mission_state, dtype=torch.long),
            'search_state': torch.tensor(search_state, dtype=torch.long)
        }
        

# Training function
def train_model(config, load_tokenizer=False):
    # Set random seeds for reproducibility
    print('Setting random seeds...')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    print('Creating output directory...')
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration
    print('Saving configuration...')
    with open(os.path.join(config.output_dir, 'config_language2motion_transformer.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)

    # Load dataset
    print('Loading dataset...')
    dataset_path = f'generated_vlm_dataset_n{config.n_dataset}_cxn025/vison_language_motion_pair_format_split82_n{config.n_dataset}'
    dataset = load_from_disk(dataset_path)

    # Build tokenizer
    def build_tokenizer(train_data):
        # Initialize tokenizer with word-level model
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Configure tokenizer trainer
        trainer = trainers.WordLevelTrainer(
            vocab_size=30000,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        # Generator function to provide training data
        def get_tokens():
            for item in train_data:
                input_str = (
                    f"{item['mission_instruction_0']} [SEP] {item['mission_instruction_1']} "
                    f"[SEP] {item['predicted_object']} [SEP] confidence:{item['confidence'][0]:.2f} "
                    f"[SEP] object_xyn:{item['object_xyn'][0]:.2f} {item['object_whn'][1]:.2f} "
                    f"[SEP] {item['mission_state_in']}"
                    f"[SEP] {item['search_state_in']}"
                )
                yield input_str.split()
        
        # Train tokenizer
        tokenizer.train_from_iterator(get_tokens(), trainer=trainer)
        
        # Wrap as HuggingFace tokenizer
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token='[PAD]',
            unk_token='[UNK]',
            cls_token='[CLS]',
            sep_token='[SEP]',
            mask_token='[MASK]'
        )
    
    # Load or build tokenizer
    if load_tokenizer:
        print('Loading tokenizer...')
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    else:
        print('Building tokenizer...')
        tokenizer = build_tokenizer(dataset['train'])
        tokenizer.save_pretrained(config.tokenizer_dir)

    # Create datasets
    print('Creating datasets...')
    train_dataset = VLMDataset(dataset['train'], tokenizer, config.max_seq_length)
    test_dataset = VLMDataset(dataset['test'], tokenizer, config.max_seq_length)

    # Create data loaders
    print('Creating data loaders...')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Initialize model
    print('Initializing model...')
    model = LanguageToMotionTransformer(config, tokenizer.vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    # Optimizer and loss functions
    print('Setting up optimizer and loss functions...')
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    motion_criterion = nn.MSELoss()  # For regression of motion vector
    state_criterion = nn.CrossEntropyLoss()  # For classification of states

    # Training loop
    print('Starting training loop...')
    best_test_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_motion_loss, train_mission_state_loss, train_search_state_loss = 0, 0, 0
        
        for batch in train_loader:
            # Move data to device
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            motions = batch['motion'].to(device)
            mission_states = batch['mission_state'].to(device)
            search_states = batch['search_state'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_motion, pred_mission_state, pred_search_state = model(inputs, masks)
            
            # Calculate losses
            motion_loss = motion_criterion(pred_motion, motions)
            mission_state_loss = state_criterion(pred_mission_state, mission_states)
            search_state_loss = state_criterion(pred_search_state, search_states)
            
            # Combined loss with weighted motion component
            total_loss = config.beta * motion_loss + mission_state_loss + search_state_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_motion_loss += motion_loss.item()
            train_mission_state_loss += mission_state_loss.item()
            train_search_state_loss += search_state_loss.item()
        
        # Evaluation phase
        model.eval()
        test_motion_loss, test_mission_state_loss, test_search_state_loss = 0, 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                motions = batch['motion'].to(device)
                mission_states = batch['mission_state'].to(device)
                search_states = batch['search_state'].to(device)
                
                # Forward pass
                pred_motion, pred_mission_state, pred_search_state = model(inputs, masks)
                
                # Calculate losses
                test_motion_loss += motion_criterion(pred_motion, motions).item()
                test_mission_state_loss += state_criterion(pred_mission_state, mission_states).item()
                test_search_state_loss += state_criterion(pred_search_state, search_states).item()
        
        # Calculate average losses
        avg_train_motion = train_motion_loss / len(train_loader)
        avg_train_mission = train_mission_state_loss / len(train_loader)
        avg_train_search = train_search_state_loss / len(train_loader)
        avg_train_total = config.beta * avg_train_motion + avg_train_mission + avg_train_search
        
        avg_test_motion = test_motion_loss / len(test_loader)
        avg_test_mission = test_mission_state_loss / len(test_loader)
        avg_test_search = test_search_state_loss / len(test_loader)
        avg_test_total = config.beta * avg_test_motion + avg_test_mission + avg_test_search
        
        # Save best model
        if avg_test_total <= best_test_loss:
            best_test_loss = avg_test_total
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'language2motion_transformer.pth'))
            print(f'Saved best model at Epoch {epoch + 1}')
        
        # Print epoch summary
        print(f'Epoch {epoch + 1}/{config.epochs}:\n'
              f'Train - Motion Loss: {avg_train_motion:.4f}, '
              f'Mission State Loss: {avg_train_mission:.4f}, '
              f'Search State Loss: {avg_train_search:.4f}, '
              f'Total Loss: {avg_train_total:.4f}\n'
              f'Test  - Motion Loss: {avg_test_motion:.4f}, '
              f'Mission State Loss: {avg_test_mission:.4f}, '
              f'Search State Loss: {avg_test_search:.4f}, '
              f'Total Loss: {avg_test_total:.4f}\n'
              f'----------------------------------------')
    
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Language to Motion Transformer')
    
    # Add arguments for all configuration parameters
    parser.add_argument('--n_dataset', type=int, default=1000000, help='Dataset size')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="", help='Output directory')
    parser.add_argument('--tokenizer_dir', type=str, default="", help='Tokenizer directory')
    parser.add_argument('--beta', type=int, default=10, help='Weight for motion loss')
    parser.add_argument('--load_tokenizer', action='store_true', help='Load existing tokenizer')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = ModelConfig(
        n_dataset=args.n_dataset,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        beta=args.beta
    )
    
    # Start training
    train_model(config, load_tokenizer=args.load_tokenizer)

if __name__ == '__main__':
    main()
    