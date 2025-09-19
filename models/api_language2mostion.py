# api_language2motion.py
import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
from torch import nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

@dataclass
class ModelConfig:
    n_dataset: int = 1000000
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_seq_length: int = 128
    epochs: int = 200
    seed: int = 42
    output_dir: str = f'model_language2motion_n{n_dataset}_d{d_model}_h{nhead}_l{num_layers}_f{dim_feedforward}_msl{max_seq_length}_hold_success'
    tokenizer_dir: str = f'tokenizer_language2motion_n{n_dataset}'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LanguageToMotionTransformer(nn.Module):
    def __init__(self, config, vocab_size):
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
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.motion_head = nn.Linear(config.d_model, 3)
        self.mission_state_head = nn.Linear(config.d_model, 4)
        self.search_state_head = nn.Linear(config.d_model, 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        x = x[:, 0]  # CLS token
        motion = self.motion_head(x)
        mission_state = self.mission_state_head(x)
        search_state = self.search_state_head(x)
        return motion, mission_state, search_state


class MotionPredictor:
    def __init__(self, model_path="model_language2motion", tokenizer_path="tokenizer_language2motion"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load configuration
        config_path = os.path.join(model_path, "config_language2motion_transformer.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        self.config = ModelConfig(** config_data)
        
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Initialize model
        self.model = LanguageToMotionTransformer(self.config, self.tokenizer.vocab_size)
        
        # Load weights
        model_path = os.path.join(model_path, "language2motion_transformer.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_input(self, data):
        """Convert input data to the format required by the model"""
        input_str = (
            f"{data['mission_instruction_0']} [SEP] {data['mission_instruction_1']} "
            f"[SEP] {data['predicted_object']} [SEP] confidence:{data['confidence'][0]:.2f} "
            f"[SEP] object_xyn:{data['object_xyn'][0]:.2f} {data['object_xyn'][1]:.2f} "
            f"[SEP] object_whn:{data['object_whn'][0]:.2f} {data['object_whn'][1]:.2f} "
            f"[SEP] {data['mission_state_in']}"
            f"[SEP] {data['search_state_in']}"
        )
        encoding = self.tokenizer(
            input_str,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].bool().to(self.device)
        }

    def predict(self, data):
        """Perform prediction"""
        with torch.no_grad():
            inputs = self._preprocess_input(data)
            motion, mission_state, search_state = self.model(inputs["input_ids"], inputs["attention_mask"])
            
            # Process outputs
            motion_vector = motion.cpu().numpy()[0].tolist()
            mission_state_class = torch.argmax(mission_state, dim=1).item()
            search_state_class = torch.argmax(search_state, dim=1).item()
            mission_state_map = {0: "success", 1: "searching", 2: "researching", 3: "running"}
            search_state_map = {0: "had_searching", 1: "had_researching"}
            
            return {
                "motion_vector": [round(v, 2) for v in motion_vector],
                "predicted_state": mission_state_map[mission_state_class],
                "search_state": search_state_map[search_state_class]
            }

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = MotionPredictor(model_path="model_language2motion_n1000000_d128_h8_l4_f512_msl64_hold_success", tokenizer_path="tokenizer_language2motion_n1000000")
    
    # Example input ground truth
    sample_input = {"mission_instruction_0":"Make haste to auto at -0.05 m\/s","mission_object_0":"car","mission_instruction_1":"Make haste to cutlery at 0.82 m\/s","mission_object_1":"knife","predicted_object":"knife","confidence":[0.5354537108],"object_xyn":[0.6707287949,0.22678588],"object_whn":[0.5876534978,0.862209081],"mission_state_in":"running","search_state_in":"had_searching_1","motion_vector":[0.82,0.0,-0.3414575898],"mission_state_out":"searching_1","search_state_out":"had_searching_1"}

    # sample_input = {"mission_instruction_0":"Make way to avian at the specified speed of -0.11 m\/s","mission_object_0":"bird","mission_instruction_1":"Make way to clutch at the specified speed of 1.19 m\/s","mission_object_1":"handbag","predicted_object":"handbag","confidence":[0.824352656],"object_xyn":[0.2312953662,0.3076264599],"object_whn":[0.456865857,0.0039395511],"mission_state_in":"searching_0","search_state_in":"had_searching_0","motion_vector":[0.0,0.0,-0.3],"mission_state_out":"searching_1","search_state_out":"had_searching_1"}

    
    # Perform prediction
    prediction = predictor.predict(sample_input)
    print("Prediction results:")
    print(f"Motion vector: {prediction['motion_vector']}")
    print(f"Predicted state: {prediction['predicted_state']}")
    print(f"Search state: {prediction['search_state']}")
    