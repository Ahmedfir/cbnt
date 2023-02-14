import logging
import os
import sys
from enum import Enum
from os.path import isfile
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from cb.code_bert_mlm import CodeBertMlmFillMask, MAX_TOKENS, MASK, ListCodeBertPrediction
from cb.job_config import JobConfig
from cb.predict_json_locs import surround_method, cut_method
from cb.replacement_mutants import FileReplacementMutants, DetailedReplacementMutant
from utils.assertion_utils import is_empty_strip
from utils.file_read_write import load_file

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))


class CodePosition(BaseModel):
    startPosition: int = None
    endPosition: int = None


class Location(BaseModel):
    node: str = None
    codePosition: CodePosition = None
    nodeType: str = None
    firstMutantId: int = None
    operator: str = None
    suffix: str = None
    predictions: ListCodeBertPrediction = None
    original_token: str = None

    def get_pred_req(self, file_string, method_start, method_end, method_tokens, cbm,
                     method_before_tokens, method_after_tokens, max_size: int = MAX_TOKENS):
        code_position = self.codePosition
        start = code_position.startPosition
        end = code_position.endPosition
        self.original_token = file_string[start: end + 1]
        masked_method_string = file_string[method_start: start] + MASK + file_string[end + 1: method_end + 1]
        masked_method_tokens = cbm.tokenize(masked_method_string)
        if method_before_tokens is not None and method_after_tokens is not None:
            masked_method_tokens = method_before_tokens + masked_method_tokens + method_after_tokens
        if len(masked_method_tokens) > max_size:
            start_cutting_index, masked_method_tokens = cut_method(masked_method_tokens, max_size,
                                                                   int(max_size / 3), MASK)
            original_method_tokens = method_tokens[start_cutting_index: max_size + start_cutting_index]
        else:
            original_method_tokens = method_tokens
        assert len(masked_method_tokens) <= max_size
        return self.original_token, masked_method_tokens, original_method_tokens, self.suffix

    def set_predictions(self, predictions: ListCodeBertPrediction):
        self.predictions = predictions
        self.predictions.add_mutant_id(self.firstMutantId)


class LineLocations(BaseModel):
    line_number: int = None
    cos_func: str = 'scipy'
    # fixme make sure the the locations are unique.
    locations: List[Location] = None

    def unique_locations(self, with_preds_only=True) -> List[Location]:
        """because of an issue in spoon, often we get duplicate tokens in the return stmts.
        this should be handled early on, in the future.
        This is a temporary solution,
         but a proper discarding of the duplicated locations early on must be implemented.
         """
        # fixme make sure the the locations are unique.
        res = []
        for l in self.locations:
            if (not with_preds_only or l.predictions is not None) and not any((
                                                                                      lambda: t.codePosition.startPosition == l.codePosition.startPosition and t.codePosition.endPosition == l.codePosition.endPosition)()
                                                                              for t in res):
                res.append(l)
        return res

    @staticmethod
    def _calculate_cosine_by_loc(lp, cbm, masked_code, masked_token, suffix, original_code_tokens, max_size=MAX_TOKENS):
        tokens_arr = lp.get_original_and_predictions_tokens(cbm, masked_code, masked_token, suffix,
                                                            original_code_tokens, max_size=max_size)
        # calculate cosines
        return cbm.cosine_similarity_batch(tokens_arr)[0]

    def _batch_cosine_per_loc(self, cbm, masked_codes, reqs, locs_preds, add_cosine_nosuffix, max_size=MAX_TOKENS):
        # add cosines to the predictions and predictions to locations.
        for i, location in enumerate(self.locations):  # fixme use unique_locs instead of locations
            loc_has_no_suffix = is_empty_strip(location.suffix)
            lp = locs_preds[i]
            if not lp.has_cosines():
                # calculate cosines
                cosines = self._calculate_cosine_by_loc(lp, cbm, masked_codes[i], reqs[i][0], reqs[i][3], reqs[i][2],
                                                        max_size=max_size)
                lp.add_cosine(cosines, loc_has_no_suffix)
            if add_cosine_nosuffix and not lp.has_cosine_nosufs():
                if not loc_has_no_suffix:
                    # calculate cosines
                    cosines_nosuf = self._calculate_cosine_by_loc(lp, cbm, masked_codes[i], reqs[i][0], '', reqs[i][2],
                                                                  max_size=max_size)
                    lp.add_cosine_nosuf(cosines_nosuf)
                else:
                    lp.add_cosine_nosuf_same_as_cosine()

            location.set_predictions(lp)

    def _batch_cosine_locs(self, cbm, masked_codes, reqs, locs_preds, add_cosine_nosuffix, max_size=MAX_TOKENS):
        # get tokens
        locs_pred_tokens = [
            lp.get_original_and_predictions_tokens(cbm, masked_codes[i], reqs[i][0], reqs[i][3], reqs[i][2],
                                                   max_size=max_size)
            for i, lp in enumerate(locs_preds)]

        if add_cosine_nosuffix:
            locs_pred_tokens_nosuf = [
                lp.get_original_and_predictions_tokens(cbm, masked_codes[i], reqs[i][0], '', reqs[i][2],
                                                       max_size=max_size)
                for i, lp in enumerate(locs_preds) if
                not is_empty_strip(self.locations[i].suffix)]  # fixme use unique_locs instead of locations
            if len(locs_pred_tokens_nosuf) > 0:
                locs_pred_tokens.extend(locs_pred_tokens_nosuf)

        # flatten this and batch .
        locs_pred_tokens_1d = np.concatenate(locs_pred_tokens)
        if len(locs_pred_tokens_1d) != len(locs_pred_tokens) * (cbm.predictions_number + 1):
            log.error(
                'line {0} ignored : did not rcieve {1} predictions per token'.format(str(self.line_number),
                                                                                     str(cbm.predictions_number)))
            return
        # calculate cosines
        cosines = cbm.cosine_similarity_batch(locs_pred_tokens_1d)
        if add_cosine_nosuffix and len(cosines) > len(locs_preds):
            cosines_nosuf = cosines[len(locs_preds):]
            cosines = cosines[:len(locs_preds)]

        assert len(cosines) == len(locs_preds)
        j = -1
        # add cosines to the predictions and predictions to locations.
        for i, lp in enumerate(locs_preds):
            loc_has_no_suffix = is_empty_strip(
                self.locations[i].suffix)  # fixme use unique_locations instead of locations
            lp.add_cosine(cosines[i], loc_has_no_suffix)
            if not loc_has_no_suffix and add_cosine_nosuffix:
                j = j + 1
                lp.add_cosine_nosuf(cosines_nosuf[j])
            self.locations[i].set_predictions(lp)  # fixme use unique_locations instead of locations

    def has_predictions(self):
        return all([loc.predictions is not None for loc in self.unique_locations(with_preds_only=False)])

    def job_done(self, job_config):
        return (not job_config.add_cosine or job_config.cosine_func == self.cos_func) and all(
            [loc.predictions is not None and loc.predictions.job_done(job_config) for loc in
             self.unique_locations(with_preds_only=False)]
        )

    def process_locs(self, cbm, file_string, method_start, method_end, method_tokens, method_before_tokens,
                     method_after_tokens, job_config: JobConfig, max_size=MAX_TOKENS):
        # log.info('pred : line {0}'.format(str(self.line_number)))
        # fixme make sure the the locations are unique, use unique_locations when possible.

        # get requests
        reqs = [loc.get_pred_req(file_string, method_start, method_end, method_tokens, cbm,
                                 method_before_tokens, method_after_tokens, max_size=max_size)
                for loc in self.locations]

        # predict
        masked_codes = [cbm.decode_tokens_to_str(masked_code_tokens_req[1]) for masked_code_tokens_req in reqs]

        for code in masked_codes:
            assert 0 < cbm.tokens_count(code) <= 512

        if self.has_predictions():
            log.info('skipped predictions already processed line.')
            predictions_arr_arr = [loc.predictions for loc in self.locations]
        else:
            # predicting...
            predictions_arr_arr = cbm.call_func(masked_codes)

        #  adding of the prediction matches the masked token.
        locs_preds = [
            predictions_arr_arr[i].add_match_original(reqs[i][0], reqs[i][3])
            for i in
            range(len(predictions_arr_arr)) if len(masked_codes[i]) > 0 and len(reqs[i][0]) > 0]
        # checking that nothing is missing else ignore these locs.
        # fixme make sure the the locations are unique, use unique_locations when possible.
        #  this check will have to be adapted.
        if not (len(locs_preds) == len(predictions_arr_arr) == len(reqs) == len(self.locations)):
            log.error(
                '{0} locations (tokens) are ignored in line {1} because of a missing param: '
                'masked_code or masked_token'.format(str(len(predictions_arr_arr) - len(locs_preds)),
                                                     str(self.line_number)))
            return

        if job_config.add_cosine:
            if self.cos_func != job_config.cosine_func:
                self._reset_cosines(locs_preds)
            if job_config.memory_aware:
                self._batch_cosine_per_loc(cbm, masked_codes, reqs, locs_preds, job_config.add_cosine_nosuff,
                                           max_size=max_size)
            else:
                self._batch_cosine_locs(cbm, masked_codes, reqs, locs_preds, job_config.add_cosine_nosuff,
                                        max_size=max_size)
            self.cos_func = job_config.cosine_func
        else:
            for i, lp in enumerate(locs_preds):
                # fixme make sure the the locations are unique, use unique_locations when possible.
                self.locations[i].set_predictions(lp)

    @staticmethod
    def _reset_cosines(locs_preds):
        for lp in locs_preds:
            lp.reset_cosines()


class MethodLocations(BaseModel):
    startLineNumber: int = None
    endLineNumber: int = None
    codePosition: CodePosition = None
    methodSignature: str = None
    line_predictions: List[LineLocations] = None

    def job_done(self, job_config):
        return all([loc.job_done(job_config) for loc in self.line_predictions])

    def process_locs(self, cbm: CodeBertMlmFillMask, file_string, job_config, max_size: int = MAX_TOKENS):
        # log.info('pred : method {0}'.format(self.methodSignature))
        # log.info('--- parallel {0}'.format(str(parallel)))
        if self.job_done(job_config):
            log.info('skipped already processed file {0}'.format(self.methodSignature))
            return
        method_start = self.codePosition.startPosition
        method_end = self.codePosition.endPosition
        method_string = file_string[method_start: method_end + 1]
        if len(method_string.strip()) == 0:
            log.error('Failed to load method in [ {0} , {1} ] named : {2}'.format(method_start, method_end,
                                                                                  self.methodSignature))
            return
        method_tokens = cbm.tokenize(method_string)
        method_before_tokens = None
        method_after_tokens = None
        if len(method_tokens) < max_size:
            max_tokens_to_add = max_size - len(method_tokens)
            method_before_str = file_string[max(0, method_start - max_tokens_to_add):method_start - 1]
            method_after_str = file_string[method_end + 1:min(method_end + 1 + max_tokens_to_add, len(file_string) - 1)]
            method_before_tokens = [] if len(method_before_str.strip()) == 0 else cbm.tokenize(method_before_str)
            method_after_tokens = [] if len(method_after_str.strip()) == 0 else cbm.tokenize(method_after_str)
            method_tokens, method_before_tokens, method_after_tokens = surround_method(method_tokens,
                                                                                       method_before_tokens,
                                                                                       method_after_tokens, max_size)

        for line_loc in self.line_predictions:
            line_loc.process_locs(cbm, file_string, method_start, method_end, method_tokens, method_before_tokens,
                                  method_after_tokens, job_config, max_size=max_size)


class ClassLocations(BaseModel):
    qualifiedName: str = None
    methodPredictions: List[MethodLocations] = None


class FileLocations(BaseModel):
    file_path: str = None
    classPredictions: List[ClassLocations] = None

    def get_relative_path(self, source_dir):
        return self.file_path.split(source_dir)[1]

    def job_done(self, job_config):
        return all([m.job_done(job_config) for c in self.classPredictions for m in c.methodPredictions])

    def process_locs(self, cbm, job_config, max_size=MAX_TOKENS):
        if self.job_done(job_config):
            log.info('skipped already processed file {0}'.format(self.file_path))
            return
        log.info('pred : file {0}'.format(self.file_path))
        try:
            file_string = load_file(self.file_path)
            for class_loc in self.classPredictions:
                # log.info('pred : class {0}'.format(class_loc.qualifiedName))
                method_locs = class_loc.methodPredictions
                for method_loc in method_locs:
                    method_loc.process_locs(cbm, file_string, job_config, max_size=max_size)
        except UnicodeDecodeError:
            log.exception('Failed to load file : {0}'.format(self.file_path))


class Method:

    def __init__(self, proj_bug_id, fileP, classP, methodP, version):
        self.pid_bid = proj_bug_id
        self.file = fileP.file_path
        self.rel_file = fileP.get_relative_path(proj_bug_id)
        self.version = version
        self.class_name = classP.qualifiedName
        self.method_signature = methodP.methodSignature
        self.line_start = methodP.startLineNumber
        self.line_end = methodP.endLineNumber
        self.char_start = methodP.codePosition.startPosition
        self.char_end = methodP.codePosition.endPosition


class Mutant:

    def __init__(self, proj_bug_id, id, cosine, rank, version, match_org, score, file_path: str, class_name: str,
                 method_signature: str, line: int, has_suffix, nodeType: str,
                 operator: str, start_pos: int, end_pos: int,
                 masked_on_added: bool = False, old_val: str = None, new_val: str = None,
                 node: str = None):
        self.proj_bug_id = proj_bug_id
        self.id = id
        self.cosine = cosine
        self.rank = rank
        self.version = version
        self.match_org = match_org
        self.score = score
        # getting relative path only.
        self.file_path = os.path.sep + os.path.sep.join(
            os.path.abspath(file_path).split(proj_bug_id)[1].split(os.path.sep)[
            1:]) if proj_bug_id in file_path else file_path
        self.class_name = class_name
        self.method_signature = method_signature
        self.line = line
        self.has_suffix = has_suffix
        self.nodeType = nodeType
        self.masked_on_added = masked_on_added
        self.old_val = old_val
        self.pred_token = new_val
        self.node = node
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.operator = operator


class VersionName(Enum):
    b = 0
    f = 1


class ListFileLocations(BaseModel):
    __root__: List[FileLocations] = None

    def job_done(self, job_config):
        return all([file_loc.job_done(job_config) for file_loc in self.__root__])

    def process_locs(self, cbm, job_config=JobConfig(), max_size=MAX_TOKENS):
        for file_loc in self.__root__:
            file_loc.process_locs(cbm, job_config, max_size=max_size)

    def to_methods(self, proj_bug_id, version) -> DataFrame:
        return pd.DataFrame(
            [vars(Method(proj_bug_id, fileP, classP, methodP, version))

             for fileP in self.__root__
             for classP in fileP.classPredictions
             for methodP in classP.methodPredictions
             ])

    def to_mutants(self, proj_bug_id, version, exclude_matching=True, no_duplicates=True) -> DataFrame:
        return pd.DataFrame(
            [vars(Mutant(proj_bug_id, mutant.id, mutant.cosine, mutant.rank, version, mutant.match_org, mutant.score,
                         fileP.file_path, classP.qualifiedName, methodP.methodSignature, lineP.line_number,
                         is_empty_strip(location.suffix), location.nodeType, location.operator,
                         location.codePosition.startPosition, location.codePosition.endPosition,
                         old_val=location.original_token,
                         new_val=mutant.token_str + location.suffix, node=location.node))

             for fileP in self.__root__
             for classP in fileP.classPredictions
             for methodP in classP.methodPredictions
             for lineP in methodP.line_predictions
             for location in lineP.unique_locations()
             for mutant in location.predictions.unique_preds(exclude_matching=exclude_matching,
                                                             no_duplicates=no_duplicates)])

    def get_mutant_by_id(self, include):
        if include is None:
            return self.get_mutants_to_exec(self, None)
        elif isinstance(include, (list, tuple, set)):
            include_ids = set(include)
        else:
            include_ids = {include}

        result = [DetailedReplacementMutant(lineP.line_number, location.original_token,
                                            str(location.nodeType),
                                            m.id, fileP.file_path, location.codePosition.startPosition,
                                            location.codePosition.endPosition + 1, m.token_str + location.suffix)

                  for fileP in self.__root__
                  for classP in fileP.classPredictions
                  for methodP in classP.methodPredictions
                  for lineP in methodP.line_predictions
                  for location in lineP.locations if location.predictions is not None
                  for m in location.predictions.__root__ if
                  m.id in include_ids]

        return result

    def to_mutants_versionfilter(self, version_filter, line_filter, proj_bug_id, changes: dict):
        return pd.DataFrame(
            [vars(
                Mutant(proj_bug_id, mutant.id, mutant.cosine_nosuf
                       , mutant.rank, version_filter(fileP.file_path, lineP.line_number, changes),
                       mutant.match_org_nosuf, mutant.score,
                       fileP.file_path, classP.qualifiedName, methodP.methodSignature, lineP.line_number,
                       is_empty_strip(location.suffix), location.nodeType, location.operator,
                       location.codePosition.startPosition, location.codePosition.endPosition, ))

                for fileP in self.__root__
                for classP in fileP.classPredictions
                for methodP in classP.methodPredictions
                for lineP in methodP.line_predictions if line_filter(lineP)
                for location in lineP.locations if location.predictions is not None
                for mutant in location.predictions.__root__
                if mutant.cosine_nosuf is not None and mutant.match_org_nosuf is not None])

    def last_id(self):
        return max({mutant.id
                    for fileP in self.__root__
                    for classP in fileP.classPredictions
                    for methodP in classP.methodPredictions
                    for lineP in methodP.line_predictions
                    for location in lineP.locations if location.predictions is not None
                    for mutant in location.predictions.__root__
                    })

    def get_mutants_to_exec(self, output_csv) -> List[FileReplacementMutants]:
        result = []
        if isfile(output_csv):
            already_treated_mutant_df = pd.read_csv(output_csv)
            already_treated_mutant_ids = set(already_treated_mutant_df['id'].unique())
        else:
            already_treated_mutant_ids = set()

        for fileP in self.__root__:
            mutants = [DetailedReplacementMutant(lineP.line_number, location.original_token,
                                                 str(location.nodeType),
                                                 m.id, fileP.file_path, location.codePosition.startPosition,
                                                 location.codePosition.endPosition + 1, m.token_str + location.suffix)
                       # ReplacementMutant(m.id, fileP.file_path, location.codePosition.startPosition,
                       #                      location.codePosition.endPosition + 1, m.token_str + location.suffix)

                       for classP in fileP.classPredictions
                       for methodP in classP.methodPredictions
                       for lineP in methodP.line_predictions
                       # todo get unique locations
                       for location in lineP.unique_locations(with_preds_only=True)
                       for m in
                       location.predictions.unique_preds(exclude_ids=already_treated_mutant_ids)
                       ]
            if len(mutants) > 0:
                result.append(FileReplacementMutants(fileP.file_path, mutants))

        return result
