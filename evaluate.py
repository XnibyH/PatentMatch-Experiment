import os
import sys
import mlflow
from sklearn.metrics import matthews_corrcoef, f1_score
from src.utils import timestamp
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from transformers.pipelines.text_classification import ClassificationFunction
import torch
from src.settings import (
    MLFLOW_EXPERIMENT_NAME,
    )


if __name__ == "__main__":
    print('Starting Evaluation Script...')
        
    # Check if a GPU is available and set the device
    device = 0 if torch.cuda.is_available() else -1

    # Load the Test Dataset
    print('Loading the Test Dataset')

    dataset_path = 'data/test_clean.parquet'
    df_test = pd.read_parquet(dataset_path)

    # X
    sentence_pairs = list(zip(df_test['text'].tolist(),df_test['text_b'].tolist()))
    # sentence pairs as list of dicts for transformer's pipeline
    sentence_pairs_lods = [{"text": x[0], "text_pair": x[1]} for x in sentence_pairs]

    # y_true
    labels_true = df_test['label'].tolist()

    #@title Select the Model for Evaluation
    #@markdown Make sure to fine-tune and save the base model before selecting ***_FT** models
    all_models = {
        'stsb-roberta-large': {'model': 'cross-encoder/stsb-roberta-large', 'tokenizer': 'cross-encoder/stsb-roberta-large'},
        # fine-tuned models below
        'stsb-roberta-large_FT': {'model': 'saved_models/stsb-roberta-large_FT', 'tokenizer': 'cross-encoder/stsb-roberta-large'},
    }

    for selection in ['stsb-roberta-large', 'stsb-roberta-large_FT']:
        print(f'Evaluating: {selection}')

        # check if FT model exists
        if '_FT' in selection:
            if os.path.isdir(all_models[selection]['model']):
                pass
            else:
                print(f'Model {selection} does not exist. Please run the finetuning.py script first!')
                sys.exit(1)

        # select model
        selected_model = all_models[selection]

        #Load the Model and Tokenizer
        print('Loading Model and Tokenizer.')

        # set num_labels for selected model - cross-encoder support only 1 label
        num_labels = 1
        # load model
        model = AutoModelForSequenceClassification.from_pretrained(selected_model['model'], num_labels=num_labels)
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(selected_model['tokenizer'])

        # Start Evaluation
        print('Starting Evaluation...')
        # init mlflow experiment (use existing one)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        # run experiment
        with mlflow.start_run(experiment_id=experiment.experiment_id, log_system_metrics=True) as run:
            # set run name
            mlflow.set_tag(key='mlflow.runName',
                            value=f"Test_{selected_model['model'].split('/')[1]}_{timestamp()}")

            # log parameters
            mlflow.log_params({
                'PyTorch Device': torch.cuda.get_device_name(torch.cuda.current_device()),
                'Model': selected_model['model'],
                'Dataset': dataset_path,
                'AutoModel Parameters': model,
                'Tokenizer': tokenizer,
            })

            # run pipeline for model predictions
            pipe = pipeline("text-classification",
                            model = model,
                            tokenizer = tokenizer,
                            padding = True,
                            truncation = True,
                            device = device,
                            function_to_apply = ClassificationFunction.SIGMOID,
                            top_k=1,  # return only predicted label with score
                            )

            predictions = pipe(sentence_pairs_lods)

            # compute and log metrics
            threshold = 0.50
            
            print(f'Computing metrics for threshold: {threshold}.')

            mlflow.log_metric("Threshold", threshold)
            labels_pred = [0 if x[0]['score'] <= threshold else 1 for x in predictions]

            f1_score_value = f1_score(y_true=labels_true, y_pred=labels_pred, pos_label=1, average='binary')
            mlflow.log_metric("F1 Score", f1_score_value)

            matthews_corrcoef_value = matthews_corrcoef(y_true=labels_true, y_pred=labels_pred)
            mlflow.log_metric("Matthews Correlation Coefficient", matthews_corrcoef_value)

            print(f"F1 Score: {f1_score_value}\nMatthews Correlation Coefficient: {matthews_corrcoef_value}")

        # end experiment
        mlflow.end_run()
