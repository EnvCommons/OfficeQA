"""OfficeQA — OpenReward sandbox environment for grounded document reasoning.

Agents analyze U.S. Treasury Bulletin documents (1939-2025) to answer complex
numerical questions requiring document retrieval, parsing, and multi-step
analytical reasoning. Scoring uses deterministic fuzzy numerical matching.

Paper: https://arxiv.org/abs/2603.08655
Dataset: https://github.com/databricks/officeqa
"""

import logging
import os
from pathlib import Path

import pandas as pd
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from reward import extract_final_answer, score_answer

logger = logging.getLogger(__name__)

# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    _DATA_DIR = Path("/orwd_data")
else:
    _DATA_DIR = Path(__file__).parent

_answers: dict[str, str] = {}
_train_tasks: list[JSONObject] = []
_test_tasks: list[JSONObject] = []


def _load_csv(path: Path) -> list[dict]:
    """Load a CSV and return list of row dicts."""
    if not path.exists():
        logger.warning(f"Data file not found: {path}")
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _build_tasks(records: list[dict], task_list: list[JSONObject]) -> None:
    """Build task specs from CSV records, stripping answer field."""
    for record in records:
        uid = str(record["uid"])
        _answers[uid] = str(record["answer"])

        # Parse source_files (newline-separated in CSV)
        source_files_raw = str(record.get("source_files", ""))
        source_files = [f.strip() for f in source_files_raw.split("\n") if f.strip()]

        task_list.append({
            "id": uid,
            "question": str(record["question"]),
            "difficulty": str(record.get("difficulty", "")),
            "source_files": source_files,
        })


# Load train (full) and test (pro) splits
_full_records = _load_csv(_DATA_DIR / "officeqa_full.csv")
_pro_records = _load_csv(_DATA_DIR / "officeqa_pro.csv")

_build_tasks(_full_records, _train_tasks)
_build_tasks(_pro_records, _test_tasks)

# Sort for stable ordering
_train_tasks.sort(key=lambda t: t["id"])
_test_tasks.sort(key=lambda t: t["id"])


# --- Pydantic parameter models ---

class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    answer: str


# --- Environment class ---

class OfficeQA(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        uid = str(task_spec["id"])
        if uid not in _answers:
            raise ValueError(f"Unknown task id: {uid}")

        self.uid = uid
        self.question: str = task_spec["question"]
        self.source_files: list[str] = task_spec.get("source_files", [])
        self.ground_truth: str = _answers[uid]
        self.submitted = False

        # OpenReward API key for sandbox
        or_api_key = (
            secrets.get("OPENREWARD_API_KEY")
            or secrets.get("api_key")
            or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        )
        if not or_api_key:
            raise ValueError("OpenReward API key required (pass as OPENREWARD_API_KEY)")

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/OfficeQA",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="0.5:1",
            block_network=False,  # 22% of questions require web access
            bucket_config=SandboxBucketConfig(
                mount_path="/home/ubuntu/documents",
                read_only=True,
                only_dir="officeqa/",
            ),
        )

        or_client = AsyncOpenReward(api_key=or_api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return _train_tasks
        elif split == "test":
            return _test_tasks
        return []

    async def get_prompt(self) -> list[TextBlock]:
        # Format source file hints
        if self.source_files:
            source_section = "The following Treasury Bulletin documents are most relevant to this question:\n"
            for f in self.source_files:
                source_section += f"- {f}\n"
        else:
            source_section = "No specific source documents were identified for this question. Search the full corpus.\n"

        prompt = f"""You are solving a document-based question answering task from the OfficeQA benchmark.

## Question

{self.question}

## Source Documents

{source_section}
All 696 Treasury Bulletin documents (1939-2025) are available at /home/ubuntu/documents/.
Each document is a text file named like `treasury_bulletin_YYYY_MM.txt` with Markdown-formatted tables.

## Environment

You have a Linux sandbox with Python 3.12 and common data science libraries
(pandas, numpy, scipy, sklearn, statsmodels, matplotlib).

Use the `bash` tool to search, read, and analyze documents. You can:
- Search documents: `grep -r "pattern" /home/ubuntu/documents/`
- Read specific documents: `cat /home/ubuntu/documents/treasury_bulletin_1941_01.txt`
- Run Python for computation: `python3 -c "..."`
- Write and run Python scripts for complex analysis (regression, statistical tests, etc.)
- Access the web via `curl` or `wget` for external data (e.g., CPI values, exchange rates)

## Submission

When you have your final answer, submit it using the `submit` tool.
Your answer should be precise — most answers are numerical.
- Include units if specified in the question (e.g., "1608.80%")
- For list answers, use bracket format: [value1, value2, ...]
- Be precise with decimal places as specified in the question"""

        return [TextBlock(text=prompt)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the sandbox environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your final answer for scoring. This is a terminal action — you get one attempt."""
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        # Extract from <FINAL_ANSWER> tags if present
        try:
            predicted = extract_final_answer(params.answer)
        except ValueError:
            predicted = params.answer

        reward = score_answer(self.ground_truth, predicted, tolerance=0.00)

        result_text = f"Your answer: {predicted}\nScore: {reward:.1f}"

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={"predicted": predicted, "correct": reward == 1.0},
            reward=reward,
            finished=True,
        )
