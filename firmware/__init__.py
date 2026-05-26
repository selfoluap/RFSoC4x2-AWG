from . import signals


def __getattr__(name):
    if name == "OverlayController":
        from .overlay_controller import OverlayController

        return OverlayController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OverlayController", "signals"]
