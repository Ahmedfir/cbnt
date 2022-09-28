import datetime
import logging
import sys

PRINT_DELTA_TIME = True
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))


class DeltaTime:
    def __init__(self, logging_level=logging.INFO):
        if PRINT_DELTA_TIME:
            self.start = datetime.datetime.now()
            log.setLevel(logging_level)

    def print(self, message='DeltaTime'):
        if PRINT_DELTA_TIME:
            log.debug('DTime - {0} : {1}'.format(message, datetime.datetime.now() - self.start))
