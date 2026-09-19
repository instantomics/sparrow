def posterior(inputs, outputs):
    from .api import posterior as run

    return run(inputs, outputs)


__all__ = ["posterior"]
