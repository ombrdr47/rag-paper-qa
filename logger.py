"""
Logging Configuration for Research Paper RAG System

Provides centralized logging setup with both file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOG_FILE, CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL


def setup_logging(
    log_file: str = LOG_FILE,
    console_level: str = CONSOLE_LOG_LEVEL,
    file_level: str = FILE_LOG_LEVEL,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Configure application-wide logging.
    
    Args:
        log_file: Name of the log file
        console_level: Logging level for console output (ERROR, WARNING, INFO, DEBUG)
        file_level: Logging level for file output
        log_dir: Directory for log files (default: current directory)
        
    Returns:
        Configured root logger
    """
    # Create log directory if specified
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
    else:
        log_path = Path(log_file)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(getattr(logging, file_level))
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - minimal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level))
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
