class Registry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            cls._registry[name] = subclass
            return subclass

        return wrapper

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        assert name in cls._registry, f"Class {name} is not registered."
        instance = cls._registry[name](*args, **kwargs)
        return instance

    @classmethod
    def create_all(cls, *args, **kwargs):
        instances = []
        for name in cls.list_registered():
            instances.append(cls.create(name, *args, **kwargs))
        return instances

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())


class DataRegistry(Registry):
    _registry = {}

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        assert name in cls._registry, f"Class {name} is not registered."
        instance = cls._registry[name].from_source(*args, **kwargs)
        return instance
