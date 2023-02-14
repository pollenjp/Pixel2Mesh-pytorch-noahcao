# Standard Library
from logging.config import dictConfig
from pathlib import Path


def reset_logging_config(log_file_path: Path) -> None:
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "console_formatter": {
                    "format": "".join(
                        [
                            "[%(asctime)s]",
                            "[%(name)s]",
                            "[%(levelname)s]",
                            "[%(threadName)s]",
                            "[%(processName)s]",
                            "[%(filename)s:%(lineno)4d]",
                            " - %(message)s",
                        ]
                    ),
                }
            },
            "handlers": {
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console_formatter",
                },
                "file_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{log_file_path}",
                    "mode": "a",
                    "maxBytes": 1048576,  # 1 * 1024 * 1024,
                    "backupCount": 10,
                    "encoding": "utf-8",
                    "level": "DEBUG",
                    "formatter": "console_formatter",
                },
            },
            "disable_existing_loggers": True,
            "loggers": {
                "": {},  # disable the root logger
                "__main__": {
                    "level": "DEBUG",
                    "handlers": [
                        "console_handler",
                        "file_handler",
                    ],
                },
            },
        }
    )
