"""
common.py

Utility functions for logging, versioning, and reproducibility in bioinformatics pipelines.

Author: Expert Engineer
Date: 2024-06
"""

import logging
import os
import subprocess
import sys
from datetime import datetime


def setup_logging(log_path=None, level=logging.INFO):
    """
    Set up logging for a script.
    If log_path is provided, logs are written to file as well as stdout.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_git_commit_hash():
    """
    Returns the current git commit hash, or None if not in a git repo.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return None


def get_script_version_info(script_path=None):
    """
    Returns a string with script version info: timestamp, script name, git commit hash.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    script = script_path if script_path else os.path.basename(sys.argv[0])
    git_hash = get_git_commit_hash()
    version_info = f"Script: {script}\nTimestamp: {timestamp}"
    if git_hash:
        version_info += f"\nGit commit: {git_hash}"
    else:
        version_info += "\nGit commit: N/A"
    return version_info


def save_version_info(output_dir, script_path=None):
    """
    Save version info to a VERSION.txt file in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    version_info = get_script_version_info(script_path)
    version_file = os.path.join(output_dir, "VERSION.txt")
    with open(version_file, "w") as f:
        f.write(version_info + "\n")
    return version_file


def log_environment(logger=None):
    """
    Log Python version, platform, and key package versions.
    """
    import platform
    import sys

    import pkg_resources

    lines = [
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Platform: {platform.platform()}",
    ]
    # List key packages
    packages = [
        "scanpy",
        "anndata",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "celltypist",
        "mygene",
        "pydeseq2",
        "liana",
        "upsetplot",
        "scipy",
    ]
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            lines.append(f"{pkg}: {version}")
        except Exception:
            lines.append(f"{pkg}: not installed")
    msg = "\n".join(lines)
    if logger:
        logger.info("Environment info:\n" + msg)
    else:
        print("Environment info:\n" + msg)


# Example usage (in your main script):
# from utils.common import setup_logging, save_version_info, log_environment
# logger = setup_logging("my_script.log")
# save_version_info(output_dir="outputs/")
# log_environment(logger)
