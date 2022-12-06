import difflib
import logging
import os
import sys

from utils.cmd_utils import safe_chdir

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


def clone_checkout(url, repo_dir, rev_hash):
    log.debug(os.getcwd())
    log.info('Cloning {0} into {1}'.format(url, repo_dir))
    os.system('git clone ' + url + ' ' + repo_dir)
    with safe_chdir(repo_dir):
        log.info('checking-out {0}'.format(rev_hash))
        os.system('git checkout ' + rev_hash)


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
