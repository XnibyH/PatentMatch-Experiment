from datetime import datetime


def timestamp():
    """This function creates current timestamp"""
    return datetime.now().strftime("%Y_%m_%d_%H%M_%S")

# def binarizer(model: str, predictions: list, threshold: float):  # TODO
#     """
#     Predictions binarizer for cross-encoder output from transformer text-classification pipeline.
#     """
#     pass

def binarize_logits(logits, threshold: float=0.51):
    """
    Binarizes logits from model predictions.
    logits: list of one element (depends on num_labels) lists
    """
    # sotfmax 
    softmaxed = [0,1]
    # binarization
    binarized = [0 if x <= threshold else 1 for x in softmaxed]
    return binarized
