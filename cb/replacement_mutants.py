import logging
import sys
from os import makedirs
from os.path import join, isfile, isdir
from pathlib import Path
from subprocess import TimeoutExpired
from typing import List

from utils.file_read_write import load_file, write_file

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

TESTS_TIME_OUT_RESULT = ['TESTS_TIME_OUT']


class ReplacementMutant:
    def __init__(self, id: int, file_path: str,
                 start: int, end: int, replacement: str):
        self.id = id
        self.file_path = file_path
        self.start = start
        self.end = end
        self.replacement = replacement
        self.compilable = None
        self.broken_tests = None

    def apply_mutation(self, original):
        return original[: self.start] + self.replacement + original[self.end:]

    def output_mutated_file(self, output_dir, tmp_original_file=None):
        output_file = join(output_dir, str(self.id), Path(self.file_path).name)
        if not isfile(output_file):
            if not isdir(join(output_dir, str(self.id))):
                makedirs(join(output_dir, str(self.id)))
            if tmp_original_file is None:
                tmp_original_file = load_file(self.file_path)
            mutated_file = self.apply_mutation(tmp_original_file)
            write_file(output_file, mutated_file)

    def compile_execute(self, project, tmp_original_file=None, reset=True):
        log.debug('{0} - compile_execute in {1}'.format(str(self.id), self.file_path))

        if tmp_original_file is None:
            tmp_original_file = load_file(self.file_path)
        mutated_file = self.apply_mutation(tmp_original_file)
        try:
            write_file(self.file_path, mutated_file)
            self.compilable = project.compile()
            if self.compilable:
                try:
                    self.broken_tests = project.test()
                except TimeoutExpired:
                    self.broken_tests = TESTS_TIME_OUT_RESULT
        finally:
            if reset:
                write_file(self.file_path, tmp_original_file)


class FileReplacementMutants:
    def __init__(self, file_path: str, mutants: List[ReplacementMutant]):
        self.file_path = file_path
        self.mutants = mutants
        assert len(mutants) > 0
