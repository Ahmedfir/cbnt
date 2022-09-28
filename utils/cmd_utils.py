import logging
import os
import subprocess
import sys
from contextlib import contextmanager

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

DEFAULT_TIMEOUT_S = 5 * 60
DAY_S = 24 * 60 * 60


def shell_call(cmd, enc='utf-8', timeout=DEFAULT_TIMEOUT_S):
    if timeout > 0:
        output = subprocess.run(cmd, capture_output=True, encoding=enc, shell=True, check=True, timeout=timeout)
    else:
        output = subprocess.run(cmd, capture_output=True, encoding=enc, shell=True, check=True)
    return output


# @see ifixr
def shellCallTemplate(cmd, enc='utf-8'):
    from subprocess import CalledProcessError
    from subprocess import Popen, PIPE
    import re
    try:
        with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding=enc) as p:
            output, errors = p.communicate()
            # print(output)
            if errors:
                m = re.search('unknown revision or path not in the working tree', errors)
                if not m:
                    raise CalledProcessError(errors, '-1')
    except CalledProcessError as e:
        log.warning(errors)
    return output


@contextmanager
def safe_chdir(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
