from datetime import datetime


def timestamp():
    """This function creates current timestamp"""
    return datetime.now().strftime("%Y_%m_%d%H_%M_%S")

# def binarizer(model: str, predictions: list, threshold: float):  # TODO
#     """
#     Predictions binarizer for cross-encoder output from transformer text-classification pipeline.
#     """
#     pass
