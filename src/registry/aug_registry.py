from typing import List
from src.registry.data_registry import Registry


class AugRegistry(Registry):
    _registry = {}

    @classmethod
    def create_all(cls, configs: List[dict]):
        instance_list = []
        for cfg in configs:
            instance_list.append(cls.create(name=cfg["type"], **cfg.get("params", {})))

        return instance_list


class TFAugRegistry(AugRegistry):
    _registry = {}


class AlbumentationsRegistry(AugRegistry):
    _registry = {}
