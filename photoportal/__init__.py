"""PhotoPortal application package with lazily loaded public components."""

__all__ = ["PhotoPortalApp", "CameraManager", "AppConfig", "ImageProcessor", "PhotoStorage"]


def __getattr__(name):
    modules = {
        "PhotoPortalApp": (".app", "PhotoPortalApp"),
        "CameraManager": (".camera", "CameraManager"),
        "AppConfig": (".config", "AppConfig"),
        "ImageProcessor": (".image_processor", "ImageProcessor"),
        "PhotoStorage": (".storage", "PhotoStorage"),
    }
    if name not in modules:
        raise AttributeError(name)
    from importlib import import_module
    module_name, attribute = modules[name]
    return getattr(import_module(module_name, __name__), attribute)
