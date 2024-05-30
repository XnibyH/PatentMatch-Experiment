from datetime import datetime


def timestamp():
    """This function creates current timestamp"""
    return datetime.now().strftime("%Y_%m_%d%H_%M_%S")
