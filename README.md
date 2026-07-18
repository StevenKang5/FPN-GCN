# Interpretable miRNA-Disease Association Prediction through Knowledge-Guided Graph Learning and Fuzzy Logical Reasoning

This work has been submitted to **Neural Networks** for consideration.

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

### 3. Model training

Training configurations can be modified in:

```text
config.yaml
```

Train the model:

```bash
python train_test.py
```

After that, you will find the 10-fold cross-validation results in `./res/model/`.


### 4. Evaluation

Run the following command to evaluate the model:

```bash
python main.py
```

After execution, the evaluation metrics of the model will be generated and displayed, including the performance results of 10-fold cross-validation.

During evaluation, the function:

```python
print_10_fold_res(base_dir="./res/model", flag=True)
```

is used to summarize the results from all folds.

If `flag=True`, the model with the AUC score closest to the average AUC among the 10 folds will be automatically selected and saved to:

`./res/best_model/`

This selected model is regarded as the representative model of the 10-fold cross-validation results and can be further used for downstream analysis and case studies.


### 5. Case Study

Run the case study analysis:

```bash
python case_study.py --diseaseID 192
```

Note: T2DM id is 192, malaria id is 664, TNBC id is 208


