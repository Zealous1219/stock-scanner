"""Recommended environment checker for the stock scanner project."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import subprocess
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MINIMUM_PYTHON = (3, 7)
RECOMMENDED_PYTHON = (3, 13)
REQUIRED_PACKAGES = {
    "baostock": "primary market data source",
    "pandas": "data processing dependency",
}
REQUIRED_DIRS = {
    "data": "cached historical bars",
    "output": "scan outputs",
}


def check_python_version() -> bool:
    logger.info("Checking Python version...")
    version = sys.version_info
    if (version.major, version.minor) >= MINIMUM_PYTHON:
        logger.info(
            "  OK Python %s.%s.%s meets the minimum requirement",
            version.major,
            version.minor,
            version.micro,
        )
        return True

    logger.error(
        "  FAIL Python %s.%s.%s is below the minimum supported version %s.%s",
        version.major,
        version.minor,
        version.micro,
        MINIMUM_PYTHON[0],
        MINIMUM_PYTHON[1],
    )
    return False


def check_recommended_python() -> bool:
    logger.info("Checking recommended Python version...")
    version = sys.version_info

    if (version.major, version.minor) == RECOMMENDED_PYTHON:
        logger.info(
            "  OK Python %s.%s.%s is the recommended interpreter",
            version.major,
            version.minor,
            version.micro,
        )
        return True

    logger.warning(
        "  WARN Current interpreter is Python %s.%s.%s, but this project is recommended on Python %s.%s",
        version.major,
        version.minor,
        version.micro,
        RECOMMENDED_PYTHON[0],
        RECOMMENDED_PYTHON[1],
    )
    logger.warning("     Recommended run command: py -3.13 stock-scanner.py")
    logger.warning("     Recommended install command: py -3.13 -m pip install -r requirements.txt")
    return False


def check_python_launcher() -> bool:
    logger.info("Checking Python 3.13 launcher availability...")

    if shutil.which("py") is None:
        logger.warning("  WARN Python launcher 'py' was not found on PATH")
        return False

    try:
        result = subprocess.run(
            ["py", "-3.13", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version_text = result.stdout.strip() or result.stderr.strip()
        logger.info("  OK %s", version_text)
        return True
    except Exception as exc:
        logger.warning("  WARN Unable to launch Python 3.13 with 'py -3.13': %s", exc)
        return False


def check_dependencies() -> bool:
    logger.info("Checking required packages...")
    all_ok = True

    for package_name, description in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(package_name)
            logger.info("  OK %s - %s", package_name, description)
        except ImportError:
            logger.error("  FAIL %s - not installed", package_name)
            logger.error("       Run: py -3.13 -m pip install %s", package_name)
            all_ok = False

    return all_ok


def check_directories() -> bool:
    logger.info("Checking required directories...")
    all_ok = True

    for dir_name, description in REQUIRED_DIRS.items():
        if os.path.exists(dir_name):
            if os.path.isdir(dir_name):
                logger.info("  OK %s/ - %s", dir_name, description)
            else:
                logger.error("  FAIL %s exists but is not a directory", dir_name)
                all_ok = False
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                logger.info("  OK %s/ - %s (created automatically)", dir_name, description)
            except Exception as exc:
                logger.error("  FAIL %s/ could not be created: %s", dir_name, exc)
                all_ok = False

    return all_ok


def check_write_permissions() -> bool:
    logger.info("Checking directory write permissions...")
    all_ok = True

    for dir_name in REQUIRED_DIRS:
        test_file = os.path.join(dir_name, ".write_test")
        try:
            with open(test_file, "w", encoding="utf-8") as handle:
                handle.write("test")
            os.remove(test_file)
            logger.info("  OK %s/ is writable", dir_name)
        except Exception as exc:
            logger.error("  FAIL %s/ is not writable: %s", dir_name, exc)
            all_ok = False

    return all_ok


def print_summary(environment_ok: bool) -> None:
    logger.info("=" * 60)
    if environment_ok:
        logger.info("Environment check passed.")
        logger.info("Next step:")
        logger.info("  Run: py -3.13 stock-scanner.py")
    else:
        logger.error("Environment check failed. Please fix the issues above and run again.")
        raise SystemExit(1)
    logger.info("=" * 60)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Stock scanner runtime environment check")
    logger.info("=" * 60)

    environment_ok = True

    if not check_python_version():
        environment_ok = False

    check_recommended_python()
    check_python_launcher()

    if not check_dependencies():
        environment_ok = False

    if not check_directories():
        environment_ok = False

    if not check_write_permissions():
        environment_ok = False

    print_summary(environment_ok)


if __name__ == "__main__":
    main()
