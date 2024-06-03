import mlflow
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from src.utils import timestamp
import torch
import numpy as np
from src.settings import (
    MLFLOW_EXPERIMENT_NAME,
    )


def logits_to_binary(logits: np.array, threshold: float=0.50) -> list:
    """
    Convert model output logits to probabilities using the sigmoid function and binarize on set threshold

    Args:
    logits (torch.Tensor or np.ndarray): Logits output from the model.
    threshold (float): threshold for binarization default 0.50

    Returns:
    binary_predictions: binary predictions list
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    
    # convert logits to probabilities (0...1)
    probabilities = torch.sigmoid(logits)

    # Binarize the output using the threshold
    binary_predictions = [0 if x <= threshold else 1 for x in probabilities]

    return binary_predictions

def compute_metrics(eval_pred) -> dict:
    """
    Function to compute custom metrics: f1 score and matthews correlation.

    Args:
    eval_pred: output from the evaluated model

    Returns:
    dictionary with F1 Score and Matthews Correlation (keys: "f1" and "mcc")
    """
    # Load metrics
    f1_metric = evaluate.load("f1")
    mcc_metric = evaluate.load("matthews_correlation")

    # eval predictions
    logits, labels = eval_pred

    # convert logits to binary predictions
    predictions = logits_to_binary(logits)

    # compute the metrics
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    mcc = mcc_metric.compute(predictions=predictions, references=labels)

    return {
        "f1": f1["f1"],
        "mcc": mcc["matthews_correlation"]
    }


# Select the Model for Fine-Tuning
all_models = {
    'stsb-roberta-large': {'model': 'cross-encoder/stsb-roberta-large', 'tokenizer': 'cross-encoder/stsb-roberta-large'},
}

selection = 'stsb-roberta-large'
selected_model = all_models[selection]


# Set mlflow parameters and start the experiment
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# # mlflow.start_run(experiment_id=experiment.experiment_id, log_system_metrics=True)
mlflow.set_tag(key='mlflow.runName', value=f"Training_{selected_model['model'].split('/')[1]}_{timestamp()}")


# Load and Train/Validation Split the Dataset
# loading train and test datasets
dataset = datasets.load_dataset("parquet", data_files={"train": "data/train_clean.parquet", "test": "data/test_clean.parquet"})

# split train into train and validation sets 20%
train_test_split = dataset['train'].train_test_split(test_size=0.20)

# rename temporary test from train for validation
train_test_split['validation'] = train_test_split.pop('test')

# full dataset: train, validation and test
dataset = datasets.DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['validation'],
    'test': dataset['test']
})

# Tokenize the Dataset
# init tokenizer
tokenizer = AutoTokenizer.from_pretrained(selected_model['tokenizer'])

def preprocess_function(batch):
    """
    Function to preprocess the dataset.
    Extends the dataset with joined and tokenized pair of sentences for model input.
    """
    # Tokenize the pairs of texts
    inputs = tokenizer(
        batch['text'], batch['text_b'],
        padding='max_length',
        truncation=True,
        max_length=tokenizer.model_max_length,  # None == tokenizer.model_max_length
        return_tensors="pt",
        )
    inputs['label'] = batch['label']
    return inputs

# preprocess the data
tokenized_dataset = dataset.map(preprocess_function, batched=True)


#Configure a Model
# set num_labels for selected model - cross-encoder support only 1 label
num_labels = 1
# init the model
model = AutoModelForSequenceClassification.from_pretrained(selected_model['model'], num_labels=num_labels)

# Set Training Arguments and Initialize Trainer
training_args = TrainingArguments(
    output_dir=f"./fine_tuning_results/{selected_model['model'].split('/')[1]}",
    num_train_epochs=5,
    per_device_train_batch_size=32,  # RTX 3090: 32
    per_device_eval_batch_size=128,  # RTX 3090: 128
    warmup_steps=20,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    learning_rate=2e-5,  # learning rate
    save_total_limit=5,  # limit the total amount of checkpoints, delete the older checkpoints
    logging_dir=f"./fine_tuning_results/{selected_model['model'].split('/')[1]}/logs",  # directory for storing logs
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
)

# init trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Start Training
# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# end experiment
mlflow.end_run()

# Save the Model
trainer.save_model(f"./saved_models/{selected_model['model'].split('/')[1]}_FT")
