import logging
import unittest

import torch

from cb import CodeBertMlmFillMask
from cb.code_bert_mlm import MASK


class MyTestCase(unittest.TestCase):

    def test_encoder_batch__2_seq(self):
        code1 = "if (real == 0.0 && imaginary == 0.0) { return NaN; }"
        code2 = "if (real == 0.0 && imaginary == 0.0) { return INF; } "
        cbm = CodeBertMlmFillMask()
        code1_tokens = cbm.tokenize(code1)
        code2_tokens = cbm.tokenize(code2)
        from utils.delta_time_printer import DeltaTime
        delta_time = DeltaTime(logging_level=logging.DEBUG)
        code1_embed = cbm.get_context_embed(code1_tokens)
        code2_embed = cbm.get_context_embed(code2_tokens)
        delta_time.print('2 separate')
        batch_embeddings = cbm.get_context_embed_batch([code1_tokens, code2_tokens])
        delta_time.print('2 batched')
        torch.equal(code1_embed, batch_embeddings[0])
        torch.equal(code2_embed, batch_embeddings[1])

    def test_encoder_decoder_1line(self):
        code = "if (real == 0.0 && imaginary == 0.0) { return NaN; }"
        cbm = CodeBertMlmFillMask()
        tokens_ids = cbm.tokenize_str_to_tokenids(code)
        decoded_code = cbm.decode_tokenids_to_str(tokens_ids)
        self.assertEqual(code, decoded_code)

    def test_encoder_decoder_1line_masked(self):
        code = "if (real == 0.0 " + MASK + "imaginary == 0.0) { return NaN; }"
        cbm = CodeBertMlmFillMask()
        tokens_ids = cbm.tokenize_str_to_tokenids(code)
        decoded_code = cbm.decode_tokenids_to_str(tokens_ids)
        self.assertEqual(code, decoded_code)

    def test_encoder_decoder_2lines(self):
        code = "if (real == 0.0 && imaginary == 0.0) {\n return NaN; \n}"
        cbm = CodeBertMlmFillMask()
        tokens_ids = cbm.tokenize_str_to_tokenids(code)
        decoded_code = cbm.decode_tokenids_to_str(tokens_ids)
        self.assertEqual(code, decoded_code)

    def test_comments_tokenizer(self):
        code = "// blalbla\n" \
               "/** \n" \
               "* comments/" \
               "* sdgsg\n" \
               "*/" \
               "public void func()"
        from cb.code_bert_mlm import CodeBertCosineEmbed
        cbm = CodeBertCosineEmbed()
        tokens = cbm.tokenize(code)
        self.assertTrue("/**" in tokens)
        self.assertTrue("*/" in tokens)

    def test_encoder_decoder_2lines_masked(self):
        code = "if (real == 0.0 && imaginary == 0.0) {\n return " + MASK + "; \n}"
        cbm = CodeBertMlmFillMask()
        tokens_ids = cbm.tokenize_str_to_tokenids(code)
        decoded_code = cbm.decode_tokenids_to_str(tokens_ids)
        self.assertEqual(code, decoded_code)


if __name__ == '__main__':
    unittest.main()
