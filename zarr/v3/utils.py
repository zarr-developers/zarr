import inspect
from contextlib import contextmanager


@contextmanager
def nested_run():
    from trio._core._run import GLOBAL_RUN_CONTEXT

    s = object()
    task, runner, _dict = s, s, s
    if hasattr(GLOBAL_RUN_CONTEXT, "__dict__"):
        _dict = GLOBAL_RUN_CONTEXT.__dict__
    if hasattr(GLOBAL_RUN_CONTEXT, "task"):
        task = GLOBAL_RUN_CONTEXT.task
        del GLOBAL_RUN_CONTEXT.task
    if hasattr(GLOBAL_RUN_CONTEXT, "runner"):
        runner = GLOBAL_RUN_CONTEXT.runner
        del GLOBAL_RUN_CONTEXT.runner

    try:
        yield
    finally:
        if task is not s:
            GLOBAL_RUN_CONTEXT.task = task
        elif hasattr(GLOBAL_RUN_CONTEXT, "task"):
            del GLOBAL_RUN_CONTEXT.task

        if runner is not s:
            GLOBAL_RUN_CONTEXT.runner = runner
        elif hasattr(GLOBAL_RUN_CONTEXT, "runner"):
            del GLOBAL_RUN_CONTEXT.runner

        if _dict is not s:
            GLOBAL_RUN_CONTEXT.__dict__.update(_dict)


class _Meta(type):
    def __init__(cls, *args, **kwargs):
        cls._sync = None

    @property
    def sync(cls):
        if cls._sync:
            return cls._sync
        cls._sync = cls._syncify()
        return cls._sync


class AutoSync(metaclass=_Meta):
    @classmethod
    def _syncify(cls, *args, **kwargs):
        class SyncSubclass(cls):
            pass

        attrs = [c for c in dir(cls) if c.startswith("async_")]
        for attr in attrs:
            meth = getattr(cls, attr)
            if inspect.iscoroutinefunction(meth):

                def cl(meth):
                    def sync_version(self, *args, **kwargs):
                        """
                        Automatically generated synchronous  version of {attr}

                        See {attr} documentation.
                        """
                        import trio

                        with nested_run():
                            return trio.run(meth, self, *args)

                    sync_version.__doc__ = (
                        "Automatically generated sync"
                         "version of {}.\n\n{}".format(attr, meth.__doc__)
                    )
                    return sync_version

                setattr(SyncSubclass, attr[6:], cl(meth))

        return SyncSubclass
