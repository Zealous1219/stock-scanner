"""
策略配置加载器

从JSON或YAML文件加载策略配置。

功能:
    - 加载策略配置
    - 验证配置有效性
    - 策略参数验证

作者: AI Assistant
日期: 2026-03-28
"""

import logging
import json
import os
from copy import deepcopy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'strategy': {
        'name': 'black_horse',
        'params': {}
    },
    'stock_pool': {
        'type': 'hs300'
    },
    'data': {
        'lookback_days': 180,
        'initial_days': 400,
        'request_interval': 0.5
    }
}


class ConfigLoader:
    """
    配置加载器

    从文件或字典加载配置，支持JSON格式。
    """

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器

        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}

        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        从文件加载配置

        参数:
            config_path: 配置文件路径

        返回值:
            配置字典
        """
        self.config_path = config_path
        _, ext = os.path.splitext(config_path)

        if ext.lower() == '.json':
            self.config = self._load_json(config_path)
        elif ext.lower() in ['.yaml', '.yml']:
            self.config = self._load_yaml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")

        logger.info(f"配置已加载: {config_path}")
        return self.config

    def _load_json(self, config_path: str) -> Dict[str, Any]:
        """加载JSON配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML未安装，将回退到JSON格式")
            raise ImportError("请安装PyYAML: pip install pyyaml")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        参数:
            key: 配置键，支持点号分隔的路径，如 'strategy.name'
            default: 默认值

        返回值:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_strategy_config(self) -> Dict[str, Any]:
        """
        获取策略配置

        返回值:
            策略配置字典
        """
        return self.get('strategy', DEFAULT_CONFIG['strategy'])

    def get_stock_pool(self) -> str:
        """
        获取股票池配置

        返回值:
            股票池类型
        """
        return self.get('stock_pool.type', DEFAULT_CONFIG['stock_pool']['type'])

    def get_data_config(self) -> Dict[str, Any]:
        """
        获取数据配置

        返回值:
            数据配置字典
        """
        return self.get('data', DEFAULT_CONFIG['data'])

    def validate(self) -> bool:
        """
        验证配置有效性

        返回值:
            True表示配置有效

        异常:
            ValueError: 配置无效
        """
        if 'strategy' not in self.config:
            raise ValueError("配置必须包含 'strategy' 字段")

        strategy = self.config['strategy']
        if 'name' not in strategy:
            raise ValueError("策略配置必须包含 'name' 字段")

        return True


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件

    参数:
        config_path: 配置文件路径，默认为 'config.json'

    返回值:
        配置字典
    """
    if config_path is None:
        config_path = 'config.json'

    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return deepcopy(DEFAULT_CONFIG)

    loader = ConfigLoader(config_path)
    config = deepcopy(DEFAULT_CONFIG)
    config.update(loader.config)
    config['strategy'] = {**DEFAULT_CONFIG['strategy'], **loader.config.get('strategy', {})}
    config['strategy']['params'] = {
        **DEFAULT_CONFIG['strategy'].get('params', {}),
        **loader.config.get('strategy', {}).get('params', {})
    }
    config['stock_pool'] = {**DEFAULT_CONFIG['stock_pool'], **loader.config.get('stock_pool', {})}
    config['data'] = {**DEFAULT_CONFIG['data'], **loader.config.get('data', {})}
    return config
