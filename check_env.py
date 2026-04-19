"""
环境依赖检测工具

功能说明:
    检查运行股票筛选工具所需的环境是否满足要求。

检测内容:
    1. Python依赖包是否已安装
    2. 必需的目录结构是否存在
    3. 目录是否可写

使用方法:
    python check_env.py

作者: AI Assistant
日期: 2026-03-27
"""

import logging
import os
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

RECOMMENDED_PYTHON = (3, 13)


def check_python_version():
    """
    检查Python版本

    说明:
        确保Python版本 >= 3.7
    """
    logger.info("检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        logger.info(f"  ✅ Python {version.major}.{version.minor}.{version.micro} - 版本满足要求")
        return True
    else:
        logger.error(f"  ❌ Python {version.major}.{version.minor}.{version.micro} - 需要 Python 3.7+")
        return False


def check_recommended_python():
    """
    检查当前是否运行在推荐的 Python 3.13 上。
    """
    logger.info("检查推荐的 Python 版本...")

    version = sys.version_info
    if (version.major, version.minor) == RECOMMENDED_PYTHON:
        logger.info(
            f"  OK Python {version.major}.{version.minor}.{version.micro} - 当前就是推荐版本"
        )
        return True

    logger.warning(
        f"  WARN 当前 Python 为 {version.major}.{version.minor}.{version.micro}，"
        f"推荐使用 Python {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}"
    )
    logger.warning("     推荐运行: py -3.13 stock-scanner.py")
    logger.warning("     推荐安装依赖: py -3.13 -m pip install -r requirements.txt")
    return False


def check_python_launcher():
    """
    检查系统是否能直接调用 Python 3.13。
    """
    logger.info("检查 Python 3.13 启动器...")

    if shutil.which("py") is None:
        logger.warning("  WARN 未找到 py 启动器，无法快捷切换到 Python 3.13")
        return False

    try:
        result = subprocess.run(
            ["py", "-3.13", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"  OK {result.stdout.strip() or result.stderr.strip()}")
        return True
    except Exception as exc:
        logger.warning(f"  WARN 无法通过 py -3.13 调用 Python 3.13: {exc}")
        return False


def check_dependencies():
    """
    检查Python依赖包

    说明:
        检查运行程序所需的第三方包是否已安装
    """
    logger.info("检查Python依赖包...")

    required_packages = {
        'baostock': '百度开源股票数据接口(主数据源)',
        'pandas': '数据处理库(必需)',
    }

    all_ok = True
    for package_name, description in required_packages.items():
        try:
            __import__(package_name)
            logger.info(f"  ✅ {package_name} - {description}")
        except ImportError:
            logger.error(f"  ❌ {package_name} - 未安装!")
            logger.error(f"     请运行: pip install {package_name}")
            all_ok = False

    return all_ok


def check_directories():
    """
    检查并创建目录结构

    说明:
        检查程序所需的目录是否存在，不存在则自动创建
    """
    logger.info("检查目录结构...")

    required_dirs = {
        'data': '股票历史数据缓存目录',
        'output': '筛选结果输出目录',
    }

    all_ok = True
    for dir_name, description in required_dirs.items():
        if os.path.exists(dir_name):
            if os.path.isdir(dir_name):
                logger.info(f"  ✅ {dir_name}/ - {description}")
            else:
                logger.error(f"  ❌ {dir_name} - 存在但不是目录!")
                all_ok = False
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"  ✅ {dir_name}/ - {description} (已自动创建)")
            except Exception as e:
                logger.error(f"  ❌ {dir_name}/ - 创建失败: {e}")
                all_ok = False

    return all_ok


def check_write_permissions():
    """
    检查目录写入权限

    说明:
        测试程序所需目录是否可写
    """
    logger.info("检查目录写入权限...")

    dirs_to_check = ['data', 'output']
    all_ok = True

    for dir_name in dirs_to_check:
        test_file = os.path.join(dir_name, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"  ✅ {dir_name}/ - 可写入")
        except Exception as e:
            logger.error(f"  ❌ {dir_name}/ - 无法写入: {e}")
            all_ok = False

    return all_ok


def print_summary(all_ok):
    """
    打印检测结果摘要
    """
    logger.info("=" * 60)
    if all_ok:
        logger.info("🎉 环境检测通过! 所有依赖和目录正常。")
        logger.info("")
        logger.info("下一步:")
        logger.info("  运行: python stock-scanner.py")
        logger.info("=" * 60)
    else:
        logger.error("❌ 环境检测未通过! 请修复上述问题后重新运行。")
        logger.error("=" * 60)
        sys.exit(1)


def main():
    """
    主函数 - 环境检测入口
    """
    logger.info("=" * 60)
    logger.info("股票筛选工具 - 运行环境检测")
    logger.info("=" * 60)
    logger.info("")

    all_ok = True

    # 检查Python版本
    if not check_python_version():
        all_ok = False

    logger.info("")

    # 检查依赖包
    if not check_dependencies():
        all_ok = False

    logger.info("")

    # 检查目录结构
    if not check_directories():
        all_ok = False

    logger.info("")

    # 检查写入权限
    if not check_write_permissions():
        all_ok = False

    logger.info("")

    # 打印结果
    print_summary(all_ok)


if __name__ == "__main__":
    main()
