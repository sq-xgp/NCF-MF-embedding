"""Offline numerical and training checks for the handwritten BERT."""
import unittest

import torch
from torch.nn import functional as F
from transformers import BertConfig as HFConfig, BertForMaskedLM

from classic_bert_from_scratch import BertConfig, ClassicBertForMLM, mask_tokens


class ClassicBertTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = BertConfig(vocab_size=31, hidden_size=16, num_layers=2,
                                 num_heads=4, intermediate_size=32, max_positions=16, dropout=0)
        self.model = ClassicBertForMLM(self.config).eval()

    def test_logits_match_transformers_with_identical_weights(self):
        config = HFConfig(vocab_size=31, hidden_size=16, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=32, max_position_embeddings=16,
                          hidden_dropout_prob=0, attention_probs_dropout_prob=0, layer_norm_eps=1e-12)
        config._attn_implementation = "eager"
        reference = BertForMaskedLM(config).eval()

        def copy_layer(source, target):
            target.load_state_dict(source.state_dict())

        copy_layer(self.model.bert.embeddings.word, reference.bert.embeddings.word_embeddings)
        copy_layer(self.model.bert.embeddings.position, reference.bert.embeddings.position_embeddings)
        copy_layer(self.model.bert.embeddings.segment, reference.bert.embeddings.token_type_embeddings)
        copy_layer(self.model.bert.embeddings.norm, reference.bert.embeddings.LayerNorm)
        for source, target in zip(self.model.bert.layers, reference.bert.encoder.layer):
            for name in ("query", "key", "value"):
                copy_layer(getattr(source.attention, name), getattr(target.attention.self, name))
            copy_layer(source.attention.output, target.attention.output.dense)
            copy_layer(source.attention_norm, target.attention.output.LayerNorm)
            copy_layer(source.ffn[0], target.intermediate.dense)
            copy_layer(source.ffn[2], target.output.dense)
            copy_layer(source.output_norm, target.output.LayerNorm)
        copy_layer(self.model.mlm_transform[0], reference.cls.predictions.transform.dense)
        copy_layer(self.model.mlm_transform[2], reference.cls.predictions.transform.LayerNorm)
        with torch.no_grad():
            reference.cls.predictions.bias.copy_(self.model.mlm_bias)
            ids = torch.tensor([[2, 6, 7, 3, 0], [2, 8, 9, 10, 3]])
            mask = ids.ne(0).long()
            segments = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
            actual = self.model(ids, mask, segments)
            expected = reference(input_ids=ids, attention_mask=mask, token_type_ids=segments).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_short_batch_masks_content_and_updates_parameters(self):
        class Tokenizer:
            mask_token_id = 4

            def __len__(self):
                return 31

        ids = torch.tensor([[2, 6, 3, 0]])
        attention = torch.tensor([[1, 1, 1, 0]])
        segments = torch.zeros_like(ids)
        special = torch.tensor([[1, 0, 1, 1]])
        original = ids.clone()
        inputs, labels = mask_tokens((ids, attention, segments, special), Tokenizer())
        torch.testing.assert_close(ids, original)
        self.assertEqual(labels.tolist(), [[-100, 6, -100, -100]])
        before = self.model.mlm_bias.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()
        logits = self.model(**inputs)
        loss = F.cross_entropy(logits.reshape(-1, 31), labels.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.mlm_bias))
        with self.assertRaises(ValueError):
            mask_tokens((ids, attention, segments, torch.ones_like(ids)), Tokenizer())


if __name__ == "__main__":
    unittest.main()
