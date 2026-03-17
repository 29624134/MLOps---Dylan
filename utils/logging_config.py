import logging
import os
import sys
from typing import Optional
from datetime import datetime


class LoggerFactory:
    """
    Centralized logging configuration factory.
    Ensures consistent logging across all modules.
    """

    _configured = False
    _loggers = {}

    @classmethod
    def configure_root_logger(
            cls,
            level: int = logging.INFO,
            log_dir: Optional[str] = None,
            console: bool = True,
            file: bool = True,
    ):
        """
        Configure the root logger once for the entire application.

        Args:
            level: Logging level (logging.INFO, logging.DEBUG, etc.)
            log_dir: Directory for log files (default: "logs")
            console: Whether to log to console
            file: Whether to log to file
        """
        if cls._configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler (UTF-8 encoded to support Unicode characters on Windows)
        if console:
            console_stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
            console_handler = logging.StreamHandler(stream=console_stream)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if file:
            if log_dir is None:
                log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, f"workflow_{timestamp}.log")

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        cls._configured = True
        logging.info(f"Logging configured: level={logging.getLevelName(level)}, console={console}, file={file}")

    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific module.

        Args:
            name: Logger name (usually __name__)
            log_file: Optional separate log file for this logger

        Returns:
            Configured logger instance
        """
        # Ensure root logger is configured
        if not cls._configured:
            cls.configure_root_logger()

        # Return cached logger if exists
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)

        # Add separate file handler if requested
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def get_step_logger(cls, step_name: str, run_id: str) -> logging.Logger:
        """
        Get a logger for a specific workflow step.
        Creates a separate log file for the step.

        Args:
            step_name: Name of the workflow step
            run_id: Workflow run ID

        Returns:
            Logger instance with step-specific file handler
        """
        logger_name = f"step.{step_name}.{run_id}"
        log_file = f"logs/{run_id}/{step_name}.log"
        return cls.get_logger(logger_name, log_file)


# Convenience functions
def setup_logging(level: int = logging.INFO, log_dir: Optional[str] = None):
    """Configure logging for the application."""
    LoggerFactory.configure_root_logger(level=level, log_dir=log_dir)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return LoggerFactory.get_logger(name)