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


def print_patch(from_version: str, to_version: str, output_patch_file):
    from utils.git_utils import make_patch
    patch = make_patch(from_version, to_version)
    write_file(output_patch_file, patch)


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

    def output_mutated_file(self, mutant_classes_output_dir, tmp_original_file=None, mutated_file=None, patch_diff=False,
                            java_file=False):
        assert patch_diff or java_file, "You need to chose at least one of these formats: patch_diff or java_file"
        output_file = join(mutant_classes_output_dir, str(self.id), Path(self.file_path).name)
        patches_output_dir = mutant_classes_output_dir + '_patches'
        output_patch_file = join(patches_output_dir, str(self.id) + Path(self.file_path).name + '.patch')
        if java_file and not isfile(output_file):
            if not isdir(join(mutant_classes_output_dir, str(self.id))):
                makedirs(join(mutant_classes_output_dir, str(self.id)))
            if mutated_file is None:
                if tmp_original_file is None:
                    tmp_original_file = load_file(self.file_path)
                mutated_file = self.apply_mutation(tmp_original_file)

            write_file(output_file, mutated_file)
        if patch_diff and not isfile(output_patch_file):
            if not isdir(patches_output_dir):
                makedirs(patches_output_dir)
            if mutated_file is None:
                if tmp_original_file is None:
                    tmp_original_file = load_file(self.file_path)
                mutated_file = self.apply_mutation(tmp_original_file)
            print_patch(tmp_original_file, mutated_file, output_patch_file)

    def compile_execute(self, project, mutant_classes_output_dir, tmp_original_file=None, reset=True, patch_diff=False,
                        java_file=False):
        log.debug('{0} - compile_execute in {1}'.format(str(self.id), self.file_path))

        if tmp_original_file is None:
            tmp_original_file = load_file(self.file_path)
        mutated_file = self.apply_mutation(tmp_original_file)
        try:
            write_file(self.file_path, mutated_file)
            self.compilable = project.compile()
            if self.compilable:
                if (patch_diff or java_file) and mutant_classes_output_dir is not None:
                    self.output_mutated_file(mutant_classes_output_dir, mutated_file=mutated_file,
                                             tmp_original_file=tmp_original_file,
                                             patch_diff=patch_diff, java_file=java_file)
                try:
                    self.broken_tests = project.test()
                except TimeoutExpired:
                    self.broken_tests = TESTS_TIME_OUT_RESULT
        finally:
            if reset:
                write_file(self.file_path, tmp_original_file)


class DetailedReplacementMutant(ReplacementMutant):
    def __init__(self, line: int, original_value: str, node_type: str, *args, **kargs):
        super(DetailedReplacementMutant, self).__init__(*args, **kargs)
        self.line = line
        self.original_value = original_value
        self.node_type = node_type


class FileReplacementMutants:
    def __init__(self, file_path: str, mutants: List[ReplacementMutant]):
        self.file_path = file_path
        self.mutants = mutants
        assert len(mutants) > 0
