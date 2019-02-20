"""
实验管理相关代码
"""

import sys
from contextlib import contextmanager
from traceback import print_tb
import yaml


@contextmanager
def capture_all_exception(_run):
    """Capture all Errors and Exceptions, print traceback and flush stdout stderr."""
    try:
        yield None

    except Exception:
        exc_type, exc_value, trace = sys.exc_info()
        print(exc_type, exc_value, trace)
        print_tb(trace)
        # _run._stop_heartbeat()
        # _run._emit_failed(exc_type, exc_value, trace.tb_next)
        # raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def yaml_load(data_path):
    with open(data_path) as f:
        return yaml.load(f)


def yaml_dump(data, data_path):
    with open(data_path, 'w') as f:
        yaml.dump(data, f)
