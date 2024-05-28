# Experiment title 

The corresponding <a href:=https://www.notion.so/Plan-of-Experiment-PoE-template-efed4153dd7849c5979e9abb00293ec0>Plan of Experiment is provided here</a>.
\
The <a href:=https://www.notion.so/Experiment-Report-Template-450e66b444c74039bd1beda4f6c226a9>Full Report</a> describing the effort is also available.

## Goal of the experiment
Do provide project goal from the PoE document.

## A long section about how to run the code, examples of use, requirements, and similar.
Do provide a bullet-proof description so that a person who never used this code will know how to get started.

```
git clone https://github.com/XnibyH/PatentMatch-Experiment.git
cd PatentMatch-Experiment
```

Also, state python version, and the compute environment where the code was executed (Ubuntu 22.04.LTS at a local machine, AWS EC2, and similar).

## Data

Reproduce the data processing:
- Download `train.parquet` and `test.parquet` to **data** dir from [Google Drive Folder](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link).
- Run the [data_exploration notebook](notebooks/data_exploration.ipynb).

## Data Processing

Data Processing as shown in [data_exploration notebook](notebooks/data_exploration.ipynb) involved:
- Fixing indexing based on *unnamed_col* and *index* columns
- Dropping NaNs in columns' subset: *updated_index*/*index*, *text*, *text_b*, *label*
- Dropping duplicated rows based on columns' subset: *text*, *text_b*
- Saving new files in *data* folder: **train_clean.parquet** and **test_clean.parquet**
- Further **Train** data split for **Train** and **Validation** sets during a model training

## Dataset Card

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

## Columns Description

Table 2. shows description of the columns from the Dataset

| Column Name           | Descritption  |
|---                    | ---           |
| index                 | index number  |
| claim_id              | id of claim from patent application|
| patent_application_id | id of patent application  |
| cited_document_id     | id of cited document      |
| text                  | claim text |
| text_b                | cited paragraph text |
| label                 | 0 - non-novelty-destroying (“A” documents, negative samples); 1 - novelty-destroying (“X” documents, positive samples) |
| date                  | date          |
| DIznQ_0               | DIznQ_0       |

## Credentials

Create .env file:

```shell
MLFLOW_TRACKING_URI=https://mlflow.dev.humandev.org/
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```
