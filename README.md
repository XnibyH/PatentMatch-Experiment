# Experiment title 

The corresponding <a href:=https://www.notion.so/Plan-of-Experiment-PoE-template-efed4153dd7849c5979e9abb00293ec0>Plan of Experiment is provided here</a>.
\
The <a href:=https://www.notion.so/Experiment-Report-Template-450e66b444c74039bd1beda4f6c226a9>Full Report</a> describing the effort is also available.

## Goal of the experiment
Do provide project goal from the PoE document.

## A long section about how to run the code, examples of use, requirements, and similar.
Do provide a bullet-proof description so that a person who never used this code will know how to get started.

```
Specifically, do provide a section with information how to copy the repository.
git clone git@github.com:HumanDevIP/Experiment-repository-template.git
```

Also, state python version, and the compute environment where the code was executed (Ubuntu 22.04.LTS at a local machine, AWS EC2, and similar).

## Data
Data downloaded from [Google Drive Folder](https://drive.google.com/drive/folders/1bReauP_LtdzBFpCk82RL3N8hvufGSr8r?usp=drive_link).

## Dataset Card

Data consists of two dataframes **train** and **test**.

## Data Processing

Data Processing involved:
- removing rows with empty cells

## Columns Description

| Column Name           | Descritption  |
|---                    | ---           |
| index                 | index number  |
| claim_id              | claim         |
| patent_application_id | id            |
| cited_document_id     | id            |
| text                  | content       |
| text_b                | content b     |
| label                 | label         |
| date                  | date          |
| DIznQ_0               | 0 - , 1 -     |

## Credentials

Create .env file:

```shell
MLFLOW_TRACKING_URI=https://mlflow.dev.humandev.org/
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```
