<div align="center">
 
# LOVON: Legged Open-Vocabulary Object Navigator</a>

[![Project Page](https://img.shields.io/badge/Project%20Page-6DE1D2?style=for-the-badge&logo=safari&labelColor=555555)](https://daojiepeng.github.io/LOVON/)
[![arXiv](https://img.shields.io/badge/arXiv-F75A5A?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2507.06747)
[![Video](https://img.shields.io/badge/Video-Bilibili-FFD63A?style=for-the-badge&logo=bilibili&labelColor=555555)](https://www.bilibili.com/video/BV1xh3ezJEJn/?share_source=copy_web&vd_source=49611dfae89f996cc7959e0fb4a4dcd3)

<p align="center">
  <img src="assets/teaser.png" width="600">
</p>

</div>


## 💡 Introduction


LOVON, a novel framework that integrates large language models (LLMs) for hierarchical task planning with open-vocabulary visual detection models, tailored for effective long-range object navigation in dynamic, unstructured environments.

<p align="center">
  <img src="assets/overveiw.png" width="600">
</p>


## TODO List

We are currently working on organizing the code for the LOVON project, and it will be released progressively. The upcoming tasks include:

- [x] **More Details About Dataset Generation**: Provide additional information on dataset generation.

- [x] **Training Details**: Provide detailed information about the model training process, including configurations, hyperparameters, and training procedures.
- [ ] **Deployment Details**: Provide guides about the deployment of the system on robots like Unitree Go2/H1-2/B2 using Jetson Orin. 


## Preface

Welcome to LOVON, a framework for training and deploying models that bridge natural language instructions with robotic motion and object perception. This guide walks you through dataset generation, model inference with pretrained examples, and training the core components: the Language-to-Motion Model (L2MM) and the Instruction Object Extractor (IOE). Finally, you will be able to deploy the trained policy on robots like Unitree Go2/H1-2/B2.

## 1. Dataset Generation 

Navigate to the project's `scripts` directory and run the data generation script. Use the `--num_samples` flag to specify the number of samples you want to generate.

> Note: To ensure high-quality data templates, we utilize Large Language Models (LLMs) for optimization—this includes refining template structure, relevance, and flexibility. The finalized, improved templates are stored in the `scripts/templates/` directory; these optimized templates then serve as the foundation for generating data that is both broadly generalizable and practically applicable.

```bash
cd ~/LOVON/scripts/
python dataset_generation.py --num_samples 1000000
```
### Output Details
The generated data is saved in the **current directory** (i.e., ~/LOVON/scripts/) with the parent folder name:
* generated_vlm_dataset_n{num_samples}_cxn025

(where {num_samples} is replaced by the value you set, e.g., 1000000).


**Quick-View Samples**: Small sample files are provided to inspect the data format without opening the full dataset. They are saved alongside the main dataset and named:

* vision_language_motion_pair_format_n{num_samples}_examples.csv
* vision_language_motion_pair_format_n{num_samples}.json



**Full Dataset Structure**:
* Unsplitted data:
scripts/generated_vlm_dataset_n{num_samples}_cxn025/vision_language_motion_pair_format_n{num_samples}/
* Train-test split (8:2 ratio):
scripts/generated_vlm_dataset_n{num_samples}_cxn025/vision_language_motion_pair_format_n{num_samples}/

## 2. Try Out the Pretained Models Examples

In the `~/LOVON/models/` directory, there are pretained model examples together with the corresponding APIs. Run the following commands to test them:

**Test Language-to-Motion Model (L2MM)**
```bash
cd ~/LOVON/models/
# try the L2MM
python api_language2motion.py
```
**Test Instruction Object Extractor (IOE)**
```bash
cd ~/LOVON/models/
# try the IOE
python api_object_extraction.py

```
**Expected Outputs**

After running the scripts, you will see predicted outputs similar to these:

```bash
~/LOVON/models$ python api_language2mostion.py
Prediction results:
Motion vector: [0.87, 0.0, -0.38]
Predicted state: searching
Search state: had_searching
~/LOVON/models$ python api_object_extraction.py
Input mission instruction: run to the bicycle at speed of 1.66 m/s
Predicted target object: bicycle
```


## 3. Train the Language-to-Motion Model (L2MM)

Navigate to the scripts directory and run the L2MM training script. Use flags like  `--n_dataset` to specify the source dataset size and other hyperparameters.

### Key Flags Note

* **First run:** Do NOT use --load_tokenizer (the script will automatically build a new tokenizer).
* **Subsequent runs:** Use --load_tokenizer to reuse the existing tokenizer (saves time).

```bash
cd ~/LOVON/scripts/
python language2motion_trainer.py \
    --n_dataset 1000000 \
    --d_model 128 \
    --nhead 4 \
    --batch_size 256 \
    --epochs 30 \
    --learning_rate 5e-5 \
    --beta 5 \
    --load_tokenizer
```
> Important: The tokenizer is critical for model performance. If you plan to use a pretrained L2MM, ensure you use the exact same tokenizer that was used during its training.

### Training Outputs

**Checkpoints & Configs:**
Trained model files are saved in the scripts/ directory (same as the training script) with names like:
* `model_language2motion_xxx.pth`: Best-performing model checkpoint (during training).
* `model_language2motion_xxx.json`: Training configuration details (hyperparameters, dataset info, etc.).

**Custom Paths:** Use these flags to override default paths:
* `--output_dir xxx`: Specify a custom directory to save checkpoints/configs.
* `--tokenizer_dir xxx`: Use a tokenizer from a custom directory (instead of the default).


### Training Progress
Training metrics (loss values) are printed to the terminal in real time. Example output:

```bash
...
Saved best model at Epoch 1
Epoch 1/30:
Train - Motion Loss: 0.0615, Mission State Loss: 0.8797, Search State Loss: 0.3686, Total Loss: 1.5559
Test  - Motion Loss: 0.0382, Mission State Loss: 0.6032, Search State Loss: 0.2632, Total Loss: 1.0574
----------------------------------------
Saved best model at Epoch 2
Epoch 2/30:
Train - Motion Loss: 0.0386, Mission State Loss: 0.5405, Search State Loss: 0.2319, Total Loss: 0.9653
Test  - Motion Loss: 0.0333, Mission State Loss: 0.4846, Search State Loss: 0.2031, Total Loss: 0.8541
----------------------------------------
...
```

## 4. Train the Instruction Object Extractor Model (IOE)

Navigate to the scripts directory and run the IOE training script. Use flags to specify the dataset size, hyperparameters, and tokenizer directory (critical: IOE must use the same tokenizer as L2MM).

```bash
cd ~/LOVON/scripts/

python object_extraction_trainer.py \
    --n_dataset 1000000 \
    --d_model 256 \
    --nhead 4 \
    --batch_size 256 \
    --epochs 30 \
    --learning_rate 5e-5 \
    --tokenizer_dir tokenizer_language2motion_n1000000
```

### Training Outputs

**Checkpoints & Configs**: Trained IOE files are saved in the scripts/ directory with names like:
* `model_object_extraction_xxx.pth`: Best-performing model checkpoint.
* `model_object_extraction_xxx.json`: Training configuration details.
**Custom Paths:** Use these flags to override defaults:
* `--output_dir xxx`: Custom directory for saving checkpoints/configs.
* `--tokenizer_dir xxx`: Path to the tokenizer used for L2MM (required for consistency).

### Training Progress
Training loss metrics are printed to the terminal. Example output:

```bash
...
Saved best model at Epoch 1
Epoch 1/30:
Train Loss: 0.3684
Test  Loss: 0.0105
----------------------------------------
Saved best model at Epoch 2
Epoch 2/30:
Train Loss: 0.0106
Test  Loss: 0.0088
----------------------------------------
...
```


