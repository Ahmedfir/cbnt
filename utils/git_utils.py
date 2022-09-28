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
