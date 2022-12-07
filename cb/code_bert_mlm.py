import json
import logging
import os
import sys
from functools import lru_cache
from os.path import isdir, join, isfile
from typing import List

from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForMaskedLM

from cb.predict_json_locs import FileSnippet
from utils.assertion_utils import assert_not_empty, is_empty_strip
from utils.caching_utils import list_to_tuple, tuple_to_list
from utils.delta_time_printer import DeltaTime
from utils.similarity_calcul import SizeFitter, cosine_similarity_chunk, torch_cosine

VOCAB_DIR = 'pre-trained/codebert-base-mlm'
VOCAB_FILE = join(VOCAB_DIR, 'vocab.json')
CODE_BERT_MLM_MODEL = "microsoft/codebert-base-mlm"
FILL_MASK_FUNCTION_NAME = 'fill-mask'
SPACE_TOKEN = "Ġ"
MASK = '<mask>'
MAX_TOKENS = 500
# default
PREDICTIONS_COUNT = 5
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
MAX_BATCH_SIZE = 20


class CbSizeFitter(SizeFitter):
    def __init__(self, items_arr, max_size: int = MAX_TOKENS):
        super(CbSizeFitter, self).__init__(items_arr, size=max_size, filling_item=SPACE_TOKEN)


class CodeBertModel:

    def save_pretrained(self, vocab_dir):
        self.model.save_pretrained(vocab_dir + '/')
        self.tokenizer.save_pretrained(vocab_dir + '/')

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        import collections
        vocab = collections.OrderedDict()
        f = open(vocab_file, )
        import json
        reader = json.load(f)
        for token in reader.keys():
            index = reader[token]
            token = token.encode("ascii", "ignore").decode()
            token = ''.join(token.split())
            vocab[index] = token
        f.close()
        return vocab

    def __init__(self, pretrained_model_name, vocab_dir, vocab_file):
        if not isdir(vocab_dir) or 0 == len(os.listdir(vocab_dir)) or not isfile(vocab_file):
            self.model = RobertaForMaskedLM.from_pretrained(pretrained_model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
            self.save_pretrained(vocab_dir)
        else:
            self.model = RobertaForMaskedLM.from_pretrained(vocab_dir)
            self.tokenizer = RobertaTokenizer.from_pretrained(vocab_dir)
        self.vocab_dict = self.load_vocab(vocab_file)

        # @list_to_tuple
        # @lru_cache(maxsize=2, typed=False)

    def tokenize(self, code_str):
        assert_not_empty(code_str)
        return self.tokenizer.tokenize(code_str)

        # @list_to_tuple
        # @lru_cache(maxsize=2, typed=False)

    def tokenize_str_to_tokenids(self, code_str):
        code_tokens = self.tokenize(code_str)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return tokens_ids

        # @list_to_tuple
        # @lru_cache(maxsize=2, typed=False)

    def tokens_count(self, code_str):
        return len(self.tokenize(code_str))

        # @list_to_tuple
        # @lru_cache(maxsize=2, typed=False)

    def decode_tokenids_to_str(self, code_tokens_ids) -> str:
        assert code_tokens_ids is not None and len(
            code_tokens_ids) > 0, "Wrong argument ! pass a source code to tokenize as string !"
        decoded_str = self.tokenizer.decode(code_tokens_ids)
        if MASK in decoded_str:
            decoded_str = decoded_str.replace(MASK, " " + MASK)
        return decoded_str

    @list_to_tuple
    @lru_cache(maxsize=5, typed=False)
    def decode_tokens_to_str(self, code_tokens) -> str:
        assert_not_empty(code_tokens)
        code_tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return self.decode_tokenids_to_str(code_tokens_ids)

    def get_context_embed_batch(self, code_tokens_arr):
        # dt = DeltaTime()
        assert_not_empty(code_tokens_arr)
        tokens_ids_arr = [self.tokenizer.convert_tokens_to_ids(code_tokens) for code_tokens in code_tokens_arr]
        import torch
        result = self.model(torch.tensor(tokens_ids_arr))[0]
        # dt.print('get_context_embed')
        return result

    @list_to_tuple
    @lru_cache(maxsize=8, typed=False)
    def get_context_embed(self, code_tokens):
        # dt = DeltaTime()
        assert_not_empty(code_tokens)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        import torch
        result = self.model(torch.tensor(tokens_ids)[None, :])[0]
        # dt.print('get_context_embed')
        return result

    def cosine_similarity_batch(self, list_list_tokens: List, k=PREDICTIONS_COUNT, batch_size=0):
        """
        1st list to next k lists
        :return: List of List of similarities
        """
        assert len(list_list_tokens) % (k + 1) == 0
        if batch_size > 0:
            # embed chunk by chunk
            embeddings = []
            for i in range(len(list_list_tokens))[::batch_size]:
                embeddings.extend(self.get_context_embed_batch(
                    list(list_list_tokens[i:min(i + batch_size, len(list_list_tokens) - 1)])))
        else:
            # embed all together
            embeddings = self.get_context_embed_batch(list_list_tokens)
        # fixme improve memory usage by freeing the tokens on
        # separate it to chunks: original + k predictions
        if batch_size == 0:
            chunk_size = k + 1
            chunks = [list(embeddings[i:i + chunk_size]) for i in range(len(embeddings))[::chunk_size]]
            sims = [cosine_similarity_chunk(torch_cosine, c[0], c[1:]) for c in chunks]
        else:
            chunks = [list(embeddings[i:min(i + batch_size, len(embeddings) - 1)]) for i in
                      range(1, len(embeddings))[::batch_size]]
            assert sum([len(c) for c in chunks]) == len(embeddings) - 1
            sims = [cosine_similarity_chunk(torch_cosine, embeddings[0], c) for c in chunks]
        return sims

    @tuple_to_list
    def cosine_similarity_tokens(self, code_tokens_1, code_tokens_2, max_size=MAX_TOKENS):
        assert_not_empty(code_tokens_1, code_tokens_2)
        # both tokens vectors have to be smaller or equal to max_size.
        # remove same number of tokens from the end of both sequences:
        # we don't want to have impactful differences which are not related to the mutation.
        delta_time = DeltaTime()
        same_size_token_lists = CbSizeFitter([code_tokens_1, code_tokens_2], max_size=max_size).fit()
        embeddings = self.get_context_embed_batch(same_size_token_lists)
        embed1 = embeddings[0]
        embed2 = embeddings[1]
        sim = torch_cosine(embed1, embed2)
        # simFT = cosine_similarity_mats_TF(embed1, embed2)
        # print(sim - simFT) ~ 0.006
        delta_time.print('embed and cosine similarity')
        return sim

    def cosine_similarity_sentences(self, statement1, statement2, max_size: int = MAX_TOKENS, space_token="Ġ"):
        code_tokens_1 = self.tokenize(statement1)
        code_tokens_2 = self.tokenize(statement2)
        return self.cosine_similarity_tokens(code_tokens_1, code_tokens_2, max_size, space_token)


class CodeBertFunction(CodeBertModel):

    def __init__(self, pretrained_model_name, function_name, vocab_dir, vocab_file):
        super().__init__(pretrained_model_name, vocab_dir, vocab_file)
        from transformers import pipeline
        self.pipeline_function = pipeline(function_name, model=self.model, tokenizer=self.tokenizer)

    def call_func(self, arg):
        return self.pipeline_function(arg)


class CodeBertPrediction(BaseModel):
    score: float
    token: int
    token_str: str
    match_org: bool = None
    match_org_nosuf: bool = None
    cosine_nosuf: float = None
    cosine: float = None
    rank: int = None
    id: int = None

    def add_match_org_nosuf(self, masked_token, suffix):
        self.match_org_nosuf = (is_empty_strip(
            suffix) and self.match_org) or self.token_str.strip() == masked_token.strip()

    def add_match_original(self, masked_token, suffix):
        self.match_org = (self.token_str.strip() + suffix) == masked_token.strip()

    def add_cosine_nosuf(self, cosine):
        self.cosine_nosuf = cosine

    def add_cosine(self, cosine):
        self.cosine = cosine

    def put_token_inplace(self, masked_code, suffix):
        # print("-{0}- put_token_inplace".format(self.token_str + suffix))
        return masked_code.replace(MASK, self.token_str + suffix)

    def job_done(self, job_config):
        return (not job_config.add_cosine or self.cosine is not None) and (
                not job_config.add_match_orig_nosuff or self.match_org_nosuf is not None) and (
                       not job_config.add_cosine_nosuff or self.cosine_nosuf is not None)


class ListCodeBertPrediction(BaseModel):
    __root__: List[CodeBertPrediction]

    def unique_preds(self, exclude_ids=None, exclude_matching=True, no_duplicates=True) -> List[CodeBertPrediction]:
        res = []
        for m in self.__root__:
            if (not exclude_matching or not m.match_org) and (exclude_ids is None or m.id not in exclude_ids) and (
                    not no_duplicates or len(res) == 0 or not any((lambda: p.token_str.strip() == m.token_str.strip())() for p in res)):
                res.append(m)
        return res

    def add_mutant_id(self, start_id=0):
        for i in range(len(self.__root__)):
            self.__root__[i].rank = i
            self.__root__[i].id = start_id + i

    def add_match_original(self, masked_token, suffix):
        for x in self.__root__:
            x.add_match_original(masked_token, suffix)
            x.add_match_org_nosuf(masked_token, suffix)
        return self

    def get_original_and_predictions_tokens(self, code_bert_func: CodeBertFunction, masked_code, masked_token, suffix,
                                            original_code_tokens, max_size=MAX_TOKENS):
        assert_not_empty(masked_code, masked_token)
        if original_code_tokens is None or len(original_code_tokens) == 0:
            original_code = masked_code.replace(MASK, masked_token)
            original_code_tokens = code_bert_func.tokenize(original_code)
        assert_not_empty(original_code_tokens)
        result = [original_code_tokens]
        for prediction in self.__root__:
            predicted_code = prediction.put_token_inplace(masked_code, suffix)
            predicted_code_tokens = code_bert_func.tokenize(predicted_code)
            result.append(predicted_code_tokens)
        return CbSizeFitter(result, max_size=max_size).fit()

    def add_cosine_nosuf_same_as_cosine(self):
        for x in self.__root__:
            x.add_cosine_nosuf(x.cosine)

    def add_cosine_nosuf(self, cosines):
        for x in self.__root__:
            assert x.cosine_nosuf is None
        for i, x in enumerate(self.__root__):
            x.add_cosine_nosuf(cosines[i])

    def add_cosine(self, cosines, no_suffix=False):
        for x in self.__root__:
            assert x.cosine is None
        for i, x in enumerate(self.__root__):
            x.add_cosine(cosines[i])
            if no_suffix:
                x.add_cosine_nosuf(cosines[i])

    def reset_cosines(self):
        for x in self.__root__:
            x.add_cosine(None)
            x.add_cosine_nosuf(None)

    def job_done(self, job_config):
        return all([pred.job_done(job_config) for pred in self.__root__])

    def has_cosines(self):
        return all([p.cosine is not None for p in self.__root__])

    def has_cosine_nosufs(self):
        return all([p.cosine_nosuf is not None for p in self.__root__])


class CodeBertCosineEmbed(CodeBertModel):

    def __init__(self):
        super().__init__(CODE_BERT_MLM_MODEL, VOCAB_DIR, VOCAB_FILE)

    def cosine_1_to_many(self, first: FileSnippet, clones: List[FileSnippet], item_to_keep=MASK,
                         batch_size=PREDICTIONS_COUNT, max_size: int = MAX_TOKENS):
        assert batch_size > 0
        batch_size = min(batch_size, len(clones))
        sims = []
        # embed chunk by chunk
        for i in range(0, len(clones))[::batch_size]:
            batch_clones_tokens = [fs.fit_max(self, max_size, item_to_keep).snippet_tokens for fs in
                                   list(clones[i:min(i + batch_size, len(clones))])]
            first_tokens = first.fit_max(self, max_size, item_to_keep).snippet_tokens
            tokens = CbSizeFitter([first_tokens] + batch_clones_tokens, max_size=max_size).fit()
            embeddings = self.get_context_embed_batch(tokens)
            sims.extend(cosine_similarity_chunk(torch_cosine, embeddings[0], embeddings[1:]))
        return sims

    def cosine_1_to_many_dep(self, first, clones, batch_size=0, max_size: int = MAX_TOKENS):
        # put in same array and make sure they are all of the same size.
        return self.cosine_similarity_batch(CbSizeFitter([first] + clones, max_size=max_size).fit(), k=len(clones),
                                            batch_size=batch_size)[0]


class CodeBertMlmFillMask(CodeBertFunction):
    def __init__(self, predictions_number=PREDICTIONS_COUNT):
        super().__init__(CODE_BERT_MLM_MODEL, FILL_MASK_FUNCTION_NAME, VOCAB_DIR, VOCAB_FILE)
        self.predictions_number = predictions_number

    def call_func(self, arg, batch_size=MAX_BATCH_SIZE):
        delta_time = DeltaTime()
        if isinstance(arg, list) and len(arg) > 1:
            batch_size = min(batch_size, len(arg))
            call_output = []
            # embed chunk by chunk
            for i in range(0, len(arg))[::batch_size]:
                batch_args = [fs for fs in
                              list(arg[i:min(i + batch_size, len(arg))])]
                call_output.extend(super().call_func(batch_args))
            result = [ListCodeBertPrediction.parse_obj(co) for co in call_output]
        else:
            call_output = super().call_func(arg)
            result = [ListCodeBertPrediction.parse_obj(call_output)]
        delta_time.print('prediction')
        return result

    # this is not used at all - i am just adding smell here...
    def __check_predictions(self, call_output):
        """
        loops through predictions and adds token_str from token if missing.
        not needed sofar.
        :param call_output:
        :return:
        """
        result = []
        for out in call_output:
            json_str = json.dumps(out)
            json_object = json.loads(json_str)
            token_str_exists = "token_str" in json_object
            if not token_str_exists:
                index = json_object["token"]
                token_str = self.vocab_dict[index]
                token_str = token_str.encode("ascii", "ignore").decode()
                json_object['token_str'] = token_str
            result.append(json_object)
        return result
