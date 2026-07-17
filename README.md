# Interpretable miRNA-Disease Association Prediction through Knowledge-Guided Graph Learning and Fuzzy Logical Reasoning

## Graphical Abstract
![The framework of FPN-GCN](framework.jpg)

## Project Structure
```text
.
├── case_study.py   
├── config.py
├── config.yaml                         # parameters configurations
├── data                                # Raw and processed datasets
│   └── final
│       ├── disease_s.csv
│       ├── hmdd_combined.csv
│       ├── miR2tar_final.csv
│       ├── miRNA_s.csv
│       └── tar2dis_final.csv
├── data_load.py                       # Data loading
├── final_train_val_test_data.json     # 10-fold cross-validation indices
├── fold_data_generation.py            # 10-fold cross-validation indices generation
├── main.py
├── model.py                           # Main architecture
├── README.md
├── res
├── train_test.py                      # Model training and evaluation
└── utils.py                           
```

## How to use

### 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the dataset and place the files under:

```text
datasets
│   └── final
│       ├── disease_s.csv
│       ├── hmdd_combined.csv
│       ├── miR2tar_final.csv
│       ├── miRNA_s.csv
│       └── tar2dis_final.csv
```

Preprocess the dataset:

```bash
python preprocess.py
```

After that, you will obtain `final_train_val_test_data.json`, which contains the k-fold cross-validation splits of the dataset. For each fold, negative samples are randomly generated for every training epoch to construct the training data dynamically.

### 3. Model training and Evaluation

Training configurations can be modified in:

```text
config.yaml
```

Train the model:

```bash
python train_test.py
```

After that, you will find the 10-fold result in ./res/model/

### 4. Case Study

Run the case study analysis:

```bash
python case_study.py
```
