"""An operation uses one immutable model/configuration snapshot."""
from contextlib import contextmanager
from contextvars import ContextVar

_profile=ContextVar('execution_model_profile',default=None)


def current_profile():
    return _profile.get()


@contextmanager
def model_context(profile):
    token=_profile.set(profile)
    try:yield profile
    finally:_profile.reset(token)
