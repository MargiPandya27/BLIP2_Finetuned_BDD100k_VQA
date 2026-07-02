import sys

from src.logger import logger


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    return (
        "Error occurred in python script name [{0}] line number [{1}] error message [{2}]"
    ).format(
        exc_tb.tb_frame.f_code.co_filename,
        exc_tb.tb_lineno,
        error,
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logger.info("Divide by zero error")
        raise CustomException(e, sys) from e
