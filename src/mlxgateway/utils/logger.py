import logging
import os
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    console = Console(highlight=False)

    def time_formatter():
        return Text(datetime.now().strftime("%H:%M:%S"), style="bold")

    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_level=True,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_extra_lines=2,
        tracebacks_show_locals=True,
    )

    rich_handler.get_time = time_formatter

    FORMAT = "%(message)s"

    logging.basicConfig(
        level=logging.NOTSET,
        format=FORMAT,
        handlers=[rich_handler],
    )

    logger_name = name if name else "mlxgateway"
    log = logging.getLogger(logger_name)

    return log


def set_logger_level(logger: logging.Logger, level: str):
    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    if level.upper() not in logging.getLevelNamesMapping():
        logger.warning(f"Invalid log level '{level}', defaulting to INFO")

    logger.setLevel(log_level)
    logging.root.setLevel(log_level)

    for handler in logging.root.handlers:
        try:
            handler.setLevel(log_level)
        except (AttributeError, TypeError):
            pass


logger = get_logger()
