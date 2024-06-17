import logging
from argparse import Namespace

from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich import box

def logging_setup(log_file):
    logger = logging.getLogger(__name__)
	
    shell_handler = RichHandler()
    file_handler = logging.FileHandler(log_file, mode='w')

    level = logging.INFO
    logger.setLevel(logging.INFO)
    shell_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    fmt_shell = "%(asctime)s %(message)s"
    fmt_file = (
        "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
    )

	
    shell_formatter = logging.Formatter(fmt_shell)
    file_formatter = logging.Formatter(fmt_file)

    # here we hook everything together
    shell_handler.setFormatter(shell_formatter)
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(level=level, handlers=[shell_handler, file_handler],
    					format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def display_config(config:Namespace, title="Configurations"):
    console = Console()
    table = Table(title=title, box=box.ROUNDED, show_lines=True, header_style="bold cyan", )
    table.add_column("Key", style="bold blue", justify="left")
    table.add_column("Value", style="green")

    for key, value in config.__dict__.items():
        table.add_row(key, str(value))
    
    console.print(table)
