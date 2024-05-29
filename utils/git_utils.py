import difflib
import logging
import os
import sys
from typing import List
from urllib.parse import urlparse

from utils.cmd_utils import safe_chdir

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def clone_checkout(url, repo_dir, rev_hash):
    if url is None or len(url) == 0 or not uri_validator(url):
        raise Exception("Clone Failed: Wrong URL.")

    log.debug(os.getcwd())
    log.info('Cloning {0} into {1}'.format(url, repo_dir))
    clone_cmd: List[str] =['git', 'clone', url, repo_dir]
    os.system(" ".join(clone_cmd))
    with safe_chdir(repo_dir):
        log.info('checking-out {0}'.format(rev_hash))
        checkout_cmd: List[str] = ['git', 'checkout', rev_hash]
        os.system(" ".join(checkout_cmd))


_no_eol = "\ No newline at end of file"


# @see https://stackoverflow.com/a/40967337/3014036
def make_patch(a, b):
    """
    Get unified string diff between two strings. Trims top two lines.
    Returns empty string if strings are identical.
    """
    diffs = difflib.unified_diff(a.splitlines(True), b.splitlines(True), n=0)
    try:
        _, _ = next(diffs), next(diffs)
    except StopIteration:
        pass
    return ''.join([d if d[-1] == '\n' else d + '\n' + _no_eol + '\n' for d in diffs])
