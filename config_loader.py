"""策略配置加载器，支持 JSON / YAML。"""

import logging
import json
import os
from copy import deepcopy
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "strategy": {"name": "black_horse", "params": {}},
    "stock_pool": {"type": "hs300"},
    "data": {"lookback_days": 180, "initial_days": 400, "request_interval": 0.5},
}


class ConfigLoader:
    """从文件加载 JSON/YAML 配置。"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        self.config_path = config_path
        _, ext = os.path.splitext(config_path)
        if ext.lower() == ".json":
            self.config = self._load_json(config_path)
        elif ext.lower() in [".yaml", ".yml"]:
            self.config = self._load_yaml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
        logger.info("配置已加载: %s", config_path)
        return self.config

    def _load_json(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML 未安装，请: pip install pyyaml")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """通过点号路径获取配置值，如 `strategy.name`。"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_strategy_config(self) -> Dict[str, Any]:
        return self.get("strategy", DEFAULT_CONFIG["strategy"])

    def get_stock_pool(self) -> str:
        return self.get("stock_pool.type", DEFAULT_CONFIG["stock_pool"]["type"])

    def get_data_config(self) -> Dict[str, Any]:
        return self.get("data", DEFAULT_CONFIG["data"])

    def validate(self) -> bool:
        if "strategy" not in self.config:
            raise ValueError("配置缺少 'strategy' 字段")
        if "name" not in self.config["strategy"]:
            raise ValueError("策略配置缺少 'name' 字段")
        return True


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件，缺失时返回默认配置。"""
    if config_path is None:
        config_path = "config.json"
    if not os.path.exists(config_path):
        logger.warning("配置文件不存在: %s，使用默认配置", config_path)
        return deepcopy(DEFAULT_CONFIG)
    loader = ConfigLoader(config_path)
    config = deepcopy(DEFAULT_CONFIG)
    config.update(loader.config)
    config["strategy"] = {**DEFAULT_CONFIG["strategy"], **loader.config.get("strategy", {})}
    config["strategy"]["params"] = {
        **DEFAULT_CONFIG["strategy"].get("params", {}),
        **loader.config.get("strategy", {}).get("params", {}),
    }
    config["stock_pool"] = {**DEFAULT_CONFIG["stock_pool"], **loader.config.get("stock_pool", {})}
    config["data"] = {**DEFAULT_CONFIG["data"], **loader.config.get("data", {})}
    return config
