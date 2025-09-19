import os
import json
import torch
from torch import nn
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass
import warnings
warnings.filterwarnings(
    "ignore",  
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.",  
    category=UserWarning,  
    module="torch.nn.modules.transformer"  
)

# Configuration class
@dataclass
class ModelConfig:
    n_dataset: int = 10000
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 512
    max_seq_length: int = 64
    epochs: int = 50
    seed: int = 42
    output_dir: str = f'model_object_extraction_n{n_dataset}_d{d_model}_h{nhead}_l{num_encoder_layers}_f{dim_feedforward}_msl{max_seq_length}_hold_success'
    tokenizer_dir: str = f'tokenizer_language2motion_n{n_dataset}'


# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Transformer model
class SequenceToSequenceClassTransformer(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        self.class_head = nn.Linear(config.d_model, num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        x = x[:, 0]  # CLS token
        logits = self.class_head(x)
        return logits


class SequenceToSequenceClassAPI:
    def __init__(self, model_path, tokenizer_path):
        # Load configuration file
        with open(os.path.join(model_path, 'config_object_extraction_transformer.json'), 'r') as f:
            config_dict = json.load(f)
        self.config = ModelConfig(**config_dict)

        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Load class mapping file
        with open(os.path.join(model_path, 'class_mapping.json'), 'r') as f:
            self.class_mapping = json.load(f)
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)

        # Initialize model
        self.model = SequenceToSequenceClassTransformer(self.config, self.tokenizer.vocab_size, self.num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load model parameters
        model_state_dict = torch.load(os.path.join(model_path, 'object_extraction_transformer.pth'), 
                                      weights_only=True,  # Load only weights
                                      map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def predict(self, mission_instruction_1):
        # Tokenize input
        encoding = self.tokenizer(
            mission_instruction_1,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Perform prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # Convert class ID to class name
        predicted_class_name = self.reverse_class_mapping[predicted_class_id]
        return predicted_class_name

# Example usage
if __name__ == "__main__":
    # Example data
    example_data = {
        "mission_instruction_0": "run to the potted plant at speed of -1.36 m/s",
        "mission_object_0": "potted plant",
        "mission_instruction_1": "run to the bicycle at speed of 1.66 m/s",
        "mission_object_1": "bicycle",
        "predicted_object": "bicycle",
        "confidence": [0.6032359135],
        "object_cxy": [0.1174651212, 0.3057501594],
        "object_xyn": [0.1174651212, 0.3057501594],
        "object_whn": [0.4610093595, 0.4246181766],
        "mission_state_in": "searching",
        "motion_vector": [0.0, 0.0, 0.3],
        "mission_state_out": "searching"
    }

    # Model path and tokenizer path
    n_dataset = 1000000
    d_model = 64
    nhead = 4
    num_encoder_layers = 2
    dim_feedforward = 256
    max_seq_length = 64
    model_path = f'model_object_extraction_n{n_dataset}_d{d_model}_h{nhead}_l{num_encoder_layers}_f{dim_feedforward}_msl{max_seq_length}_hold_success'
    tokenizer_path = f'tokenizer_language2motion_n{n_dataset}'

    # Initialize API
    api = SequenceToSequenceClassAPI(model_path, tokenizer_path)

    # Perform prediction
    mission_instruction_1 = example_data["mission_instruction_1"]
    predicted_object = api.predict(mission_instruction_1)
    print(f"Input mission instruction: {mission_instruction_1}")
    # Predicted class name
    print(f"Predicted target object: {predicted_object}")
    