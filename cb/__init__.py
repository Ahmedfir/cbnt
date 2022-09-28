import json
from os.path import isfile

from cb.code_bert_mlm import CodeBertMlmFillMask
from cb.json_locs_parser import JobConfig
from cb.json_locs_parser import ListFileLocations

PREDICTIONS_FILE_NAME = 'predictions.pickle'


def results_to_csv(result, output_file_path):
    from pandas import DataFrame
    df = DataFrame.from_dict(result)
    df.to_csv(output_file_path, encoding='utf-8', index=False, mode='a', header=not isfile(output_file_path))


# todo move to tests
def predict_masked_token(masked_code: str, masked_token, file_code_path, masked_code_line, output_file_path):
    cbm = CodeBertMlmFillMask()
    fill_mask_output = cbm.call_func(masked_code)
    fill_mask_output.add_mutant_id()
    if masked_token is not None:
        fill_mask_output.add_syntactic_similarity(cbm, masked_code, masked_token, '', None)
    predictions_list_json = fill_mask_output.json()
    result = dict()
    result['code_line'] = masked_code_line
    result['file_code_path'] = file_code_path
    result['masked_code'] = masked_code
    result['masked_token'] = masked_token
    result['predictions'] = fill_mask_output
    if output_file_path is not None:
        results_to_csv(result, output_file_path)
    else:
        print(predictions_list_json)
    return result


# todo move to tests kept only to don't refactor tests
def predict_json(masked_code_tokens_json: str, masked_token: str = None):
    assert masked_code_tokens_json is not None and len(
        masked_code_tokens_json) > 0, "Wrong argument ! pass a source code to tokenize as json !"
    print(masked_code_tokens_json)
    code_tokens_json = json.loads(masked_code_tokens_json)
    code_tokens = code_tokens_json['code_tokens']
    assert code_tokens is not None and len(
        code_tokens) > 0, "Wrong argument ! pass a source code to tokenize as json !"

    cbm = CodeBertMlmFillMask()

    masked_code = cbm.decode_tokens_to_str(code_tokens)
    fill_mask_output = cbm.call_func(masked_code)
    result = dict()
    if masked_token is not None:
        fill_mask_output.add_syntactic_similarity(cbm, masked_code, masked_token, '', None)
        result['masked_token'] = masked_token
    predictions_list_json = fill_mask_output.json()
    result['predictions'] = predictions_list_json
    return result


def predict_json_locs(sc_json_file: str, cbm: CodeBertMlmFillMask = None, job_config=JobConfig()):
    if cbm is None:
        cbm = CodeBertMlmFillMask()
    file_locs: ListFileLocations = ListFileLocations.parse_file(sc_json_file)
    print('++++++ attempt process json {0} ++++++'.format(sc_json_file))
    return predict_locs(file_locs, cbm, job_config)


def predict_locs(file_locs: ListFileLocations, cbm: CodeBertMlmFillMask = None, job_config=JobConfig()):
    if cbm is None:
        cbm = CodeBertMlmFillMask()
    print('++++++ processing ++++++')
    if file_locs.job_done(job_config):
        print('++++++ already processed ++++++')
        print('job config: {0}'.format(vars(job_config)))
    else:
        file_locs.process_locs(cbm, job_config)
    return file_locs
