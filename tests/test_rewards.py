"""Unit tests for reward functions."""

from src.training.rewards import (
    combined_reward,
    extract_code_block,
    json_reward,
    reasoning_reward,
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


class TestJsonReward:
    def test_valid_json(self):
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        assert json_reward(text) == 1.0

    def test_valid_array(self):
        text = "```json\n[1, 2, 3]\n```"
        assert json_reward(text) == 1.0

    def test_invalid_json(self):
        text = '```json\n{"name": "Alice", age: 30}\n```'
        assert json_reward(text) == 0.0

    def test_no_code_block(self):
        text = "No JSON here."
        assert json_reward(text) == 0.0

    def test_partial_credit_near_end(self):
        # Missing closing brace — error near end → 0.75
        text = '```json\n{"name": "Alice", "age": 30\n```'
        reward = json_reward(text, partial_credit=True)
        assert reward == 0.75

    def test_partial_credit_mid_error(self):
        # Error in the middle of the structure → 0.5
        text = '```json\n{"name": bad, "age": 30}\n```'
        reward = json_reward(text, partial_credit=True)
        assert reward == 0.5

    def test_partial_credit_no_json_structure(self):
        # Has a code block but content is not JSON-like → 0.25
        text = "```json\nhello world this is not json\n```"
        reward = json_reward(text, partial_credit=True)
        assert reward == 0.25

    def test_partial_credit_no_block(self):
        # No code block at all → 0.0
        text = "No JSON here."
        reward = json_reward(text, partial_credit=True)
        assert reward == 0.0

    def test_partial_credit_valid(self):
        # Valid JSON even with partial_credit → still 1.0
        text = '```json\n{"name": "Alice"}\n```'
        assert json_reward(text, partial_credit=True) == 1.0


class TestReasoningReward:
    def test_with_reasoning(self):
        text = (
            "<think>Let me think about this problem carefully"
            " and plan the solution step by step.</think>\n"
            "```json\n{}\n```"
        )
        assert reasoning_reward(text) == 0.2

    def test_without_reasoning(self):
        text = '```json\n{"key": "val"}\n```'
        assert reasoning_reward(text) == 0.0

    def test_short_reasoning(self):
        text = "<think>ok</think>\n```json\n{}\n```"
        assert reasoning_reward(text) == 0.0  # less than 20 chars


class TestCombinedReward:
    def test_json_valid(self):
        text = '```json\n{"x": 1}\n```'
        assert combined_reward(text, "json") == 1.0

    def test_json_invalid(self):
        text = "```json\n{invalid}\n```"
        assert combined_reward(text, "json") == 0.0

    def test_with_reasoning_bonus(self):
        text = (
            "<think>I need to generate a valid JSON object"
            " with the required keys and types.</think>\n"
            '```json\n{"a": 1}\n```'
        )
        reward = combined_reward(text, "json", reasoning_bonus=0.2)
        assert reward == 1.2  # 1.0 (parse) + 0.2 (reasoning)

    def test_invalid_task_type(self):
        try:
            combined_reward("text", "python")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
