# Early baseline for claim&cited-paragraph classification on PatentMatch

**Note**  
The corresponding Plan of Experiment is provided [here](https://www.notion.so/Early-baseline-for-claim-cited-paragraph-classification-on-PatentMatch-Michal-66d57dc044954503aa969fbb6edc4acc).  
The [Report](https://www.notion.so/Report-Michal-Early-Baseline-PatentMatch-Paragraph-Classification-b55cf6d528e34958947742ea152dea52) describing the effort is also available.


## Goal of the experiment

- On-hands discovery of the PatentMatch dataset, obtaining very preliminary baseline quality on the claim&cited-paragraph classification task (2 texts on model input, one binary classification label on output).

- Learn how to use the newly introduced standards for implementation of research experiments.


## Experiment Description


### Prerequisites

The experiment was conducted on a machine running **Ubuntu 22.04 LTS** with **Python 3.10.12** equipped with an **NVIDIA GeForce RTX 3090 24 GB** VRAM graphics card.  

To recreate this experiment on a similar setup machine go to [Local Experiment](#local-experiment) section. Otherwise skip to [Google Colab](#google-colab-experiment) section.


### Local Experiment

- Download this repository
```shell
git clone https://github.com/XnibyH/PatentMatch-Experiment.git
cd PatentMatch-Experiment
```

- Create venv and install experiment dependencies
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Create .env file with your credentials and variables
```
cp example.env .env
nano .env
```

- Change your credentials, save & exit
```shell
MLFLOW_TRACKING_URI= 'https://mlflow.example-server.com/'  # provide a valid mlflow server address,
MLFLOW_TRACKING_USERNAME= 'User_Name'  # your user name,
MLFLOW_TRACKING_PASSWORD= 'P455VV0RD'  # password,
MLFLOW_EXPERIMENT_NAME = 'User_Name_PatentMatchBaseline'  # and update the experiment name,
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = 'True'  # set 'True' to log system metrics: GPU utilization etc.
```

- **Download** experiment dataset: `train.parquet` and `test.parquet` to **`./data`** folder from [`Google Drive`](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link).

- Recreate processed dataset for **fine-tuning** and **testing** (described in [Data](#data) section).
```shell
python examples/recreate_dataset.py
```

- Run **fine-tuning** script for **stsb-roberta-large** model (more about selected models in [Model](#model) section).
```shell
python examples/finetune.py
```

- Run **test** script for **stsb-roberta-large** and **stsb-roberta-large *fine-tuned*** model.
```shell
python examples/evaluate.py
```

- Finally, check your MLflow server for results and metrics.

### Google Colab Experiment

To recreate the experiment in the Google Colab environment, click on the button below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XnibyH/PatentMatch-Experiment/blob/main/notebooks/colab_notebook.ipynb)


## Data

Download source files `train.parquet` and `test.parquet` from [Google Drive](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link) and save in **`./data`** folder.


### Data Processing

Data Processing as shown in [data_exploration notebook](notebooks/data_exploration.ipynb) involved:
- Fixing indexing based on *unnamed_col* and *index* columns
- Dropping NaNs in columns' subset: *updated_index*/*index*, *text*, *text_b*, *label*
- Dropping duplicated rows based on columns' subset: *text*, *text_b*, *label*
- Saving new files in *data* folder: **train_clean.parquet** and **test_clean.parquet**
- Further **Train** data split for **Train** and **Validation** sets during a model training


### Columns Description

Table 1. shows description of the columns from the Dataset

| Column Name           | Description  |
| ---                   | ---           |
| index                 | index number  |
| claim_id              | id of claim from patent application|
| patent_application_id | id of patent application  |
| cited_document_id     | id of cited document      |
| text                  | claim text    |
| text_b                | cited paragraph text |
| label                 | <table> <tbody>  <tr>  <td>0</td>  <td>non-novelty-destroying (“A” documents, negative samples)</td>  </tr>  <tr>  <td>1</td>  <td>novelty-destroying (“X” documents, positive samples)</td>  </tr>  </tbody>  </table> |
| date                  | date          |
| DIznQ_0               | DIznQ_0       |


### Dataset Card

Dataset consists of **Train** and **Test** sets and is characterized in Table 2.

| Table 2.                         | Train | Test  | 
| ---                              | ---   | ---   |
| Number of samples                | 2912  | 768   |
| Distinct patent applications     | 2346  | 597   |
| Distinct cited documents         | 2382  | 614   |
| Distinct claim texts             | 285   | 749   |
| Distinct cited paragraphs        | 289   | 766   |
| Median claim length (chars)      | 271   | 289   |
| Median paragraph length (chars)  | 479.5 | 478.5 |
| Mean claim length (chars)        | 391   | 428   |
| Mean paragraph length (chars)    | 578   | 566   |
| Non-novelty-destroying (Label 0) | 1214  | 423   |
| Novelty-destroying (Label 1)     | 1698  | 345   |


## Model

After conducting a small and somewhat limited research in search of a suitable transformer model for the current task, I decided to use a cross-encoder model and Huggingface framework. I accepted the limitations associated with the lack of embeddings for individual sentences (e.g. inefficient clustering), but I am hoping to achieve better results with Cross Encoder compared to a Sentence Transformer (a.k.a. bi-encoder) (after https://arxiv.org/abs/1908.10084).

Due to limited time and available computational resources, I chose the set of 3 models (**STSB ROBERTA BASE**, **STSB ROBERTA LARGE**, **Legal-BERT**) to illustrate the base benchmark for transformer models for conducting this experiment.

Although, to thoroughly investigate the quality of the models, I would propose, in the further steps, testing all models from the list on a selected test subset of data. This would allow for the selection of a best model for fine-tuning.

### Fine-Tuned Models Information

| Model | Size | Metrics Before Fine-Tuning | Metrics After Fine-Tuning |
| ----- | ---- | ------- | ----- |
| [STSB ROBERTA BASE](https://huggingface.co/cross-encoder/stsb-roberta-base) | 499 MB | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.472</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.070</td>  </tr>  </tbody>  </table> | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.623</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.081</td>  </tr>  </tbody>  </table> |
| [STSB ROBERTA LARGE](https://huggingface.co/cross-encoder/stsb-roberta-large ) | 1420 MB | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.544</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.004</td>  </tr>  </tbody>  </table> | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.617</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.011</td>  </tr>  </tbody>  </table> |
| [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) | 440 MB | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.603</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.027</td>  </tr>  </tbody>  </table> | <table> <tbody>  <tr>  <td>F1 Score</td>  <td>0.621</td>  </tr>  <tr>  <td>Matthews Correlation Coefficient</td>  <td>0.033</td>  </tr>  </tbody>  </table> |


### Potential Model Selection for Further Investigation

In the table below, selected models are listed along with a brief description.

| Model | Size | License | Notes |
| ----- | ---- | ------- | ----- |
| [ALL MPNET BASE V2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 438 MB | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) | Sentence-Transformer model for basic evaluation purposes. Offering best quality to embedded sentences (Performance Sentence Embeddings) and to embedded search queries & paragraphs (Performance Semantic Search) according to [Original Models table](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models). |
| [STSB ROBERTA LARGE](https://huggingface.co/cross-encoder/stsb-roberta-large ) | 1420 MB | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) | Cross-Encoder model with better performance than Bi-Encoders in sentences comparison tasks. [*](https://arxiv.org/abs/1908.10084) |
| [STSB ROBERTA BASE](https://huggingface.co/cross-encoder/stsb-roberta-base) | 499 MB | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) | Base version of previous model. |
| [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) | 440 MB | [CC-BY-SA-4.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-sa-4.0.md) | BERT model for the legal domain. |
| [EURLEX-BERT](https://huggingface.co/nlpaueb/bert-base-uncased-eurlex) | 440 MB | [CC-BY-SA-4.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-sa-4.0.md) | Sub-domain variant of Legal-BERT pre-trained on EU legislation. |
| [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) | 442 MB | [Apache 2.0](https://github.com/allenai/scibert?tab=Apache-2.0-1-ov-file#readme) | BERT model trained on scientific text. [Paper](https://arxiv.org/pdf/1903.10676) |

>Other potential models for future investigation: 
>- [BERT for Patents](https://huggingface.co/anferico/bert-for-patents)
>- [PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa) ([Arxiv Paper](https://arxiv.org/abs/2103.11933))
>- [Pegasus Big Patent](https://huggingface.co/google/pegasus-big_patent)
>- [BigBirdPegasus model (large)](https://huggingface.co/google/bigbird-pegasus-large-bigpatent)
>- [PatentSBERTa_V2](https://huggingface.co/AAUBS/PatentSBERTa_V2)
