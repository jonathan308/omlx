from __future__ import annotations

from omlx.cluster.disaggregated_worker import (
    _prefill_call_count,
    _prompt_tokens,
    _token_hash,
)


class _Tokenizer:
    bos_token_id = 7

    def encode(self, _text, *, add_special_tokens):
        assert add_special_tokens is False
        return [11, 12, 13]


def test_prompt_builder_is_exact_and_deterministic():
    assert _prompt_tokens(_Tokenizer(), 8) == [7, 11, 12, 13, 11, 12, 13, 11]
    assert _token_hash([1, 2, 3]) == _token_hash([1, 2, 3])
    assert _token_hash([1, 2, 3]) != _token_hash([1, 2, 4])


def test_prefill_call_count_matches_generate_step_frontier():
    assert _prefill_call_count(2, 2048) == 2
    assert _prefill_call_count(2048, 2048) == 2
    assert _prefill_call_count(2049, 2048) == 2
    assert _prefill_call_count(2050, 2048) == 3
