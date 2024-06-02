# Early baseline for claim&cited-paragraph classification on PatentMatch

The corresponding Plan of Experiment is provided <a href:=https://www.notion.so/Early-baseline-for-claim-cited-paragraph-classification-on-PatentMatch-Michal-66d57dc044954503aa969fbb6edc4acc>here</a>.
\
The <a href:=https://www.notion.so/Report-Michal-Early-Baseline-PatentMatch-Paragraph-Classification-b55cf6d528e34958947742ea152dea52>Report</a> describing the effort is also available.

## Goal of the experiment

- On-hands discovery of the PatentMatch dataset, obtaining very preliminary baseline quality on the claim&cited-paragraph classification task (2 texts on model input, one binary classification label on output).

- Learn how to use the newly introduced standards for implementation of research experiments.

## A long section about how to run the code, examples of use, requirements, and similar.

### Prerequisites

>performed in local environmet and in google colab

#### Run locally

#### Run in Google Colab

Also, state python version, and the compute environment where the code was executed (Ubuntu 22.04.LTS at a local machine, AWS EC2, and similar).

- Run the colab version notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XnibyH/PatentMatch-Experiment/blob/main/notebooks/finetuning_transformer_notebook.ipynb)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XnibyH/PatentMatch-Experiment/blob/main/notebooks/data_exploration.ipynb)

OR MANUALLY:

Steps to reproduce findings in this work

- Download this repository:

```shell
git clone https://github.com/XnibyH/PatentMatch-Experiment.git
cd PatentMatch-Experiment
```

- Create .env file with your credentials and variables
```
cp example.env .env
nano .env
```
```shell
MLFLOW_TRACKING_URI=
```

- Recreate the processed data:
    - Download `train.parquet` and `test.parquet` to **data** dir from [Google Drive Folder](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link).
    - Run the [data_exploration notebook](notebooks/data_exploration.ipynb).

- training: stratification ???

“The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.”

## Data

Reproduce the data processing:
- Download `train.parquet` and `test.parquet` to **data** dir from [Google Drive Folder](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link).
- Run the [data_exploration notebook](notebooks/data_exploration.ipynb).

### Data Processing

Data Processing as shown in [data_exploration notebook](notebooks/data_exploration.ipynb) involved:
- Fixing indexing based on *unnamed_col* and *index* columns
- Dropping NaNs in columns' subset: *updated_index*/*index*, *text*, *text_b*, *label*
- Dropping duplicated rows based on columns' subset: *text*, *text_b*
- Saving new files in *data* folder: **train_clean.parquet** and **test_clean.parquet**
- Further **Train** data split for **Train** and **Validation** sets during a model training

### Columns Description

Table 2. shows description of the columns from the Dataset

| Column Name           | Descritption  |
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

Data consists of **Train** and **Test** sets and is characterized in Table 1.

| Table 1.                         | Train | Test  | 
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

After conducting a small and somewhat limited research in search of a suitable transformer model for the current task, I decided to use a cross-encoder model (and Huggingface framework). I accepted the limitations associated with the lack of embeddings for individual sentences (e.g. inefficient clustering), but I am hoping to achieve better results with Cross Encoder compared to a Sentence Transformer (a.k.a. bi-encoder) (after https://arxiv.org/abs/1908.10084).

Due to limited time and available computational resources, I chose the MODEL to illustrate the base benchmark for transformer models for conducting this experiment.

Although, to thoroughly investigate the quality of the models, I would propose, in the further steps, testing all models from the list on a selected test subset of data. This would allow for the selection of a best model for fine-tuning.

### Model Selection

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

### Model Training Details
