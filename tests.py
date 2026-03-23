"""Unit tests for the OfficeQA OpenReward environment.

Tests cover:
- Reward function scoring logic
- Task structure and data integrity
- Environment class methods
"""

from pathlib import Path

import pytest

from reward import (
    extract_final_answer,
    extract_numbers_with_context,
    fuzzy_match_answer,
    score_answer,
)


# --- Reward function tests ---


class TestScoreAnswer:
    """Tests for the core score_answer function."""

    def test_exact_number(self):
        assert score_answer("2,602", "2,602") == 1.0

    def test_wrong_number(self):
        assert score_answer("2,602", "3,000") == 0.0

    def test_percentage(self):
        assert score_answer("1608.80%", "1608.80%") == 1.0

    def test_wrong_percentage(self):
        assert score_answer("1608.80%", "1500.00%") == 0.0

    def test_negative_number(self):
        assert score_answer("-184.143", "-184.143") == 1.0

    def test_decimal_precision(self):
        assert score_answer("0.096", "0.096") == 1.0

    def test_large_number_with_commas(self):
        assert score_answer("1,000,000", "1000000") == 1.0

    def test_unicode_minus(self):
        assert score_answer("−5.3", "-5.3") == 1.0

    def test_text_answer(self):
        assert score_answer("March 1977", "March 1977") == 1.0

    def test_text_wrong_month(self):
        assert score_answer("March 1977", "April 1977") == 0.0

    def test_text_in_prediction(self):
        assert score_answer("March 1977", "The answer is March 1977.") == 1.0

    def test_zero_answer(self):
        assert score_answer("0", "0") == 1.0


class TestScoreAnswerWithTolerance:
    """Tests for score_answer with non-zero tolerance."""

    def test_within_tolerance(self):
        # 2602 vs 2600: diff = 0.077%, within 1%
        assert score_answer("2,602", "2,600", tolerance=0.01) == 1.0

    def test_outside_tolerance(self):
        # 2602 vs 2500: diff = 3.9%, outside 1%
        assert score_answer("2,602", "2,500", tolerance=0.01) == 0.0


class TestListAnswers:
    """Tests for multi-number list answers."""

    def test_list_match(self):
        assert score_answer("[0.096, -184.143]", "[0.096, -184.143]") == 1.0

    def test_list_partial_mismatch(self):
        assert score_answer("[0.096, -184.143]", "[0.096, -100.0]") == 0.0


class TestExtractFinalAnswer:
    """Tests for FINAL_ANSWER tag extraction."""

    def test_with_tags(self):
        text = "Some reasoning... <FINAL_ANSWER>2,602</FINAL_ANSWER>"
        assert extract_final_answer(text) == "2,602"

    def test_without_tags(self):
        text = "2,602"
        assert extract_final_answer(text) == "2,602"

    def test_case_insensitive(self):
        text = "<final_answer>March 1977</final_answer>"
        assert extract_final_answer(text) == "March 1977"

    def test_empty_tags_raises(self):
        with pytest.raises(ValueError):
            extract_final_answer("<FINAL_ANSWER></FINAL_ANSWER>")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_final_answer("")

    def test_score_with_tags(self):
        """End-to-end: score_answer should NOT auto-extract FINAL_ANSWER tags."""
        # score_answer works on raw text; extraction happens in the submit tool
        result = score_answer("2,602", "<FINAL_ANSWER>2,602</FINAL_ANSWER>")
        # The number 2602 appears in the predicted text, so it should match
        assert result == 1.0


class TestExtractNumbers:
    """Tests for number extraction with context."""

    def test_basic_integer(self):
        result = extract_numbers_with_context("42")
        assert len(result) == 1
        assert result[0][0] == 42.0

    def test_thousands_commas(self):
        result = extract_numbers_with_context("1,000,000")
        assert len(result) == 1
        assert result[0][0] == 1000000.0

    def test_negative(self):
        result = extract_numbers_with_context("-5.3")
        assert len(result) == 1
        assert result[0][0] == -5.3

    def test_percentage(self):
        result = extract_numbers_with_context("15.5%")
        assert len(result) == 1
        assert result[0][2] is True  # has_percent

    def test_multiple_numbers(self):
        result = extract_numbers_with_context("[0.096, -184.143]")
        assert len(result) == 2


class TestFuzzyMatchRationale:
    """Tests that fuzzy_match_answer returns correct rationale."""

    def test_correct_returns_true(self):
        is_correct, rationale = fuzzy_match_answer("100", "100", tolerance=0.0)
        assert is_correct is True

    def test_incorrect_returns_false(self):
        is_correct, rationale = fuzzy_match_answer("100", "200", tolerance=0.0)
        assert is_correct is False

    def test_empty_gt_raises(self):
        with pytest.raises(ValueError):
            fuzzy_match_answer("", "100")

    def test_empty_pred_raises(self):
        with pytest.raises(ValueError):
            fuzzy_match_answer("100", "")


# --- Task structure tests ---


class TestTaskStructure:
    """Tests for data integrity. Requires prepare_data.py to have been run."""

    @pytest.fixture(autouse=True)
    def _load_data(self):
        pro_path = Path(__file__).parent / "officeqa_pro.csv"
        full_path = Path(__file__).parent / "officeqa_full.csv"
        if not pro_path.exists() or not full_path.exists():
            pytest.skip("Data files not found — run prepare_data.py first")
        import pandas as pd
        self.pro_df = pd.read_csv(pro_path)
        self.full_df = pd.read_csv(full_path)

    def test_pro_has_tasks(self):
        assert len(self.pro_df) > 0

    def test_full_has_tasks(self):
        assert len(self.full_df) > 0

    def test_full_larger_than_pro(self):
        assert len(self.full_df) >= len(self.pro_df)

    def test_required_columns(self):
        required = {"uid", "question", "answer"}
        assert required.issubset(set(self.pro_df.columns))
        assert required.issubset(set(self.full_df.columns))

    def test_unique_uids_pro(self):
        assert self.pro_df["uid"].nunique() == len(self.pro_df)

    def test_unique_uids_full(self):
        assert self.full_df["uid"].nunique() == len(self.full_df)

    def test_no_empty_questions(self):
        assert self.pro_df["question"].notna().all()
        assert self.full_df["question"].notna().all()

    def test_no_empty_answers(self):
        assert self.pro_df["answer"].notna().all()
        assert self.full_df["answer"].notna().all()


# --- Environment class tests ---


class TestOfficeQAEnv:
    """Tests for the OfficeQA environment class structure."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not (Path(__file__).parent / "officeqa_pro.csv").exists():
            pytest.skip("Data files not found — run prepare_data.py first")

    def test_list_splits(self):
        from officeqa import OfficeQA
        assert OfficeQA.list_splits() == ["train", "test"]

    def test_list_tasks_test(self):
        from officeqa import OfficeQA
        tasks = OfficeQA.list_tasks("test")
        assert len(tasks) > 0

    def test_list_tasks_train(self):
        from officeqa import OfficeQA
        tasks = OfficeQA.list_tasks("train")
        assert len(tasks) > 0

    def test_list_tasks_unknown_split(self):
        from officeqa import OfficeQA
        assert OfficeQA.list_tasks("unknown") == []

    def test_task_spec_no_answer(self):
        from officeqa import OfficeQA
        for task in OfficeQA.list_tasks("test"):
            assert "answer" not in task, f"Answer leaked in task spec for {task['id']}"

    def test_task_spec_required_fields(self):
        from officeqa import OfficeQA
        required = {"id", "question", "difficulty", "source_files"}
        for task in OfficeQA.list_tasks("test")[:5]:
            assert required.issubset(set(task.keys())), f"Missing fields in {task['id']}"

    def test_stable_ordering(self):
        from officeqa import OfficeQA
        t1 = [t["id"] for t in OfficeQA.list_tasks("test")]
        t2 = [t["id"] for t in OfficeQA.list_tasks("test")]
        assert t1 == t2
        assert t1 == sorted(t1)
