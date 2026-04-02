"""Unit tests for reward functions."""

from src.training.rewards import (
    build_reward_functions,
    combined_reward,
    extract_code_block,
    format_reward,
    reasoning_reward,
    schema_reward,
    truncation_reward,
    validity_reward,
)


class TestExtractCodeBlock:
    def test_json_fenced(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert extract_code_block(text, "json") == '{"key": "value"}'

    def test_generic_fence_fallback(self):
        text = "```\n{}\n```"
        assert extract_code_block(text, "json") == "{}"

    def test_no_fence_json_raw(self):
        text = '{"name": "test"}'
        assert extract_code_block(text, "json") == '{"name": "test"}'

    def test_no_match(self):
        text = "Just some plain text with no code."
        assert extract_code_block(text, "json") is None


class TestFormatReward:
    def test_json_fenced_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert format_reward(text) == 1.0

    def test_generic_fenced_block(self):
        text = "```\n{}\n```"
        assert format_reward(text) == 0.5

    def test_no_block(self):
        text = "No code here."
        assert format_reward(text) == 0.0


class TestValidityReward:
    def test_valid_json_object(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        assert validity_reward(text) == 1.0

    def test_valid_json_array(self):
        text = "```json\n[1, 2, 3]\n```"
        assert validity_reward(text) == 1.0

    def test_no_code_block(self):
        text = "No JSON here."
        assert validity_reward(text) == 0.0

    def test_error_near_end(self):
        # Missing closing brace — error near the end
        text = '```json\n{"name": "Alice", "age": 30\n```'
        assert validity_reward(text) == 0.70

    def test_code_block_unparseable(self):
        # Completely unparseable content — error at char 0 → "early mistake"
        text = "```json\nhello world\n```"
        assert validity_reward(text) == 0.20


class TestSchemaReward:
    def test_no_code_block(self):
        assert schema_reward("No code.", "Generate 3 items.") == 0.0

    def test_invalid_json(self):
        assert (
            schema_reward("```json\n{bad}\n```", "Generate 3 items.")
            == 0.0
        )

    def test_exact_array_count_match(self):
        text = "```json\n[1, 2, 3]\n```"
        assert (
            schema_reward(text, "Generate an array of 3 items.")
            == 1.0
        )

    def test_exact_array_count_off_by_one(self):
        text = "```json\n[1, 2]\n```"
        score = schema_reward(text, "Generate an array of 3 items.")
        assert score == 0.70

    def test_required_keys_all_present(self):
        text = '```json\n{"id": 1, "name": "x"}\n```'
        instr = 'Object with "id" (integer) and "name" (string).'
        assert schema_reward(text, instr) == 1.0

    def test_required_keys_partial(self):
        text = '```json\n{"id": 1}\n```'
        instr = 'Object with "id" (integer) and "name" (string).'
        assert schema_reward(text, instr) == 0.5

    def test_no_constraints_valid_json(self):
        text = '```json\n{"a": 1}\n```'
        assert schema_reward(text, "Generate a JSON object.") == 1.0

    def test_toplevel_array_required(self):
        text = "```json\n[1, 2]\n```"
        assert (
            schema_reward(text, "Generate a JSON array of numbers.")
            == 1.0
        )

    def test_toplevel_object_required_but_got_array(self):
        text = "```json\n[1, 2]\n```"
        assert (
            schema_reward(text, "Generate a JSON object with keys.")
            == 0.0
        )


class TestReasoningReward:
    def test_with_reasoning(self):
        text = (
            "<think>Let me think about this problem carefully"
            " and plan the solution step by step.</think>\n"
            "```json\n{}\n```"
        )
        assert reasoning_reward(text) == 1.0

    def test_without_reasoning(self):
        text = '```json\n{"key": "val"}\n```'
        assert reasoning_reward(text) == 0.0

    def test_short_reasoning(self):
        text = "<think>ok</think>\n```json\n{}\n```"
        assert reasoning_reward(text) == 0.0


class TestCombinedReward:
    def test_valid_json_no_instruction(self):
        # format=1.0, validity=1.0, schema=1.0 (no constraints), reasoning=0.0
        text = '```json\n{"x": 1}\n```'
        assert combined_reward(text) == 0.10 + 0.30 + 0.50 + 0.0

    def test_no_code_block(self):
        # format=0, validity=0, schema=0, reasoning=0
        assert combined_reward("plain text") == 0.0

    def test_with_reasoning(self):
        text = (
            "<think>I need to generate a valid JSON object"
            " with the required keys and types.</think>\n"
            '```json\n{"a": 1}\n```'
        )
        assert combined_reward(text) == 0.10 + 0.30 + 0.50 + 0.10

    def test_custom_weights(self):
        text = '```json\n{"x": 1}\n```'
        score = combined_reward(
            text,
            weight_format=0.25,
            weight_validity=0.25,
            weight_schema=0.25,
            weight_reasoning=0.25,
        )
        assert score == pytest.approx(0.25 + 0.25 + 0.25 + 0.0)


import pytest  # noqa: E402  (import after class defs to keep test grouping)


class TestTruncationReward:
    def test_fenced_complete(self):
        text = '```json\n{"key": "value"}\n```'
        assert truncation_reward(text) == 1.0

    def test_no_json_neutral(self):
        assert truncation_reward("Just plain text.") == 0.0

    def test_bare_json_complete(self):
        assert truncation_reward('{"name": "Alice"}') == 1.0

    def test_bare_json_unclosed_brace(self):
        assert truncation_reward('{"name": "Alice"') == -1.0

    def test_bare_json_unclosed_bracket(self):
        assert truncation_reward("[1, 2, 3") == -1.0

    def test_bare_json_trailing_comma(self):
        assert truncation_reward('{"a": 1,') == -1.0

    def test_bare_json_trailing_colon(self):
        assert truncation_reward('{"a":') == -1.0

    def test_bare_json_unterminated_string(self):
        assert truncation_reward('{"name": "Ali') == -1.0

    def test_bare_array_complete(self):
        assert truncation_reward("[1, 2, 3]") == 1.0


class TestBuildRewardFunctions:
    def test_returns_correct_count_with_thinking(self):
        funcs, weights = build_reward_functions(thinking=True)
        assert len(funcs) == 4
        assert len(weights) == 4
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_returns_three_without_thinking(self):
        funcs, weights = build_reward_functions(thinking=False)
        assert len(funcs) == 3
        assert len(weights) == 3
        names = [f.__name__ for f in funcs]
        assert "reasoning_reward" not in names

    def test_function_names_match_components(self):
        funcs, _ = build_reward_functions(thinking=True)
        names = [f.__name__ for f in funcs]
        assert "format_reward" in names
        assert "validity_reward" in names
        assert "schema_reward" in names
        assert "reasoning_reward" in names

    def test_each_function_returns_list(self):
        funcs, _ = build_reward_functions(thinking=True)
        completions = ['```json\n{"x": 1}\n```']
        for fn in funcs:
            result = fn(completions)
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], float)

    def test_thinking_false_redistributes_across_all_components(self):
        """When thinking=False, reasoning weight is split proportionally
        across ALL active components, including truncation."""
        funcs, weights = build_reward_functions(
            weight_format=0.25,
            weight_validity=0.30,
            weight_schema=0.30,
            weight_reasoning=0.10,
            weight_truncation=0.05,
            thinking=False,
        )
        names = [f.__name__ for f in funcs]
        assert "reasoning_reward" not in names
        assert len(funcs) == 4  # format, validity, schema, truncation
        assert pytest.approx(sum(weights), abs=1e-6) == 1.0

    def test_thinking_false_no_reasoning_weight_noop(self):
        """If reasoning weight is already 0, thinking=False changes nothing."""
        funcs, weights = build_reward_functions(
            weight_format=0.25,
            weight_validity=0.30,
            weight_schema=0.30,
            weight_reasoning=0.0,
            weight_truncation=0.15,
            thinking=False,
        )
        assert len(funcs) == 4
        assert weights == [0.25, 0.30, 0.30, 0.15]
