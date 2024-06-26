import logging

from rich.logging import RichHandler

def logging_setup(log_file: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    shell_handler = RichHandler()
    file_handler = logging.FileHandler(log_file, mode="w")

    logger.setLevel(level)
    shell_handler.setLevel(level)
    file_handler.setLevel(level)

    fmt_shell = "%(asctime)s %(message)s"
    fmt_file = (
        "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
    )

    shell_formatter = logging.Formatter(fmt_shell)
    file_formatter = logging.Formatter(fmt_file)

    # here we hook everything together
    shell_handler.setFormatter(shell_formatter)
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(
        level=level,
        handlers=[shell_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )