from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config file must contain a dictionary at root: {path}')
    return data


def merge_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    host = os.getenv('APP_HOST')
    port = os.getenv('APP_PORT')
    weights = os.getenv('MODEL_WEIGHTS')
    db_url = os.getenv('DATABASE_URL')
    redis_url = os.getenv('REDIS_URL')

    app_cfg = config.setdefault('app', {})
    model_cfg = config.setdefault('model', {})
    db_cfg = config.setdefault('database', {})
    queue_cfg = config.setdefault('queue', {})

    if host:
        app_cfg['host'] = host
    if port:
        app_cfg['port'] = int(port)
    if weights:
        model_cfg['weights'] = weights
    if db_url:
        db_cfg['url'] = db_url
    if redis_url:
        queue_cfg['redis_url'] = redis_url

    return config


def load_app_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path or os.getenv('APP_CONFIG', 'configs/app.yaml'))
    config = load_yaml(target)
    return merge_env_overrides(config)


def load_train_config(config_path: str | Path | None = None) -> dict[str, Any]:
    target = Path(config_path or os.getenv('TRAIN_CONFIG', 'configs/train.yaml'))
    return load_yaml(target)
