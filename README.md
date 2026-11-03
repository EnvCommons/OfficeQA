# OfficeQA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OfficeQA)

## Description

**OfficeQA** is an environment for evaluating AI agents on grounded, multi-document reasoning over a large corpus of U.S. Treasury Bulletins spanning nearly a century (1939–2025). Agents must retrieve relevant documents, parse dense financial tables, and perform multi-step analytical reasoning to answer precise numerical questions.

## Capabilities

- Document retrieval and search across 696 Treasury Bulletin text files
- Numerical reasoning and multi-step computation
- Statistical analysis (linear regression, geometric mean, correlation, etc.)
- Web search for external data (CPI values, exchange rates)
- Code execution for complex calculations

## Compute Requirements

- Sandbox: 0.5 CPU / 1 GB memory
- Network access enabled (required for 22% of questions)

## License

[MIT](https://opensource.org/licenses/MIT). Upstream components keep their own licences: the OfficeQA question CSVs are [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), `reward.py` is adapted from the upstream repo under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), and the Treasury Bulletin documents are U.S. Government public domain.

## Tasks

There are 246 tasks in the `train` split and 133 in `test`.

| Split | Source CSV | Count | Description |
|-------|-----------|-------|-------------|
| `train` | `officeqa_full.csv` | 246 | Full development set (easy + hard) |
| `test` | `officeqa_pro.csv` | 133 | Benchmark set (hard only) |

Each task contains a natural language question, difficulty rating, and pointers to relevant source documents.

## Reward Structure

Sparse, verifiable, binary reward (1.0 or 0.0). Scoring uses deterministic fuzzy numerical matching:

- Unit-aware comparison (million, billion, trillion, thousand)
- Configurable tolerance (default: exact match)
- Multi-number list support
- Text overlap checking for hybrid answers (e.g., "March 1977")
- `<FINAL_ANSWER>` tag extraction

The environment scores with tolerance 0 (exact numeric match after unit normalisation). A repeat submission after the answer was graded gets **-0.1**. No LLM grader is used.

## Data

- **Corpus**: 696 transformed Treasury Bulletin text files with Markdown-formatted tables
- **Source**: [databricks/officeqa](https://github.com/databricks/officeqa) on GitHub
- **Mount path**: `/home/ubuntu/documents/` in sandbox

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands: search documents, run Python, access the web |

Grading uses a hidden `@terminal` tool: when the agent is done, it replies with a plain message wrapping the final answer in `<FINAL_ANSWER>...</FINAL_ANSWER>` tags. The environment extracts the tagged answer (falling back to the whole reply if no tags) and scores it with fuzzy numeric/text matching against the reference.

## Time Horizon

Multi-turn. Agents search documents, read files, write and execute Python scripts, and optionally query external data sources before finishing with a plain-text reply.

## Environment Difficulty

In the OfficeQA Pro paper, frontier-model agents given the document corpus averaged 34.1% accuracy on the 133-question Pro set.

## Other Environment Requirements

There are no further environment requirements; OfficeQA needs no external secrets or API keys.

## Safety

Treasury Bulletin data is publicly available U.S. Government financial information. No PII or sensitive data concerns. Questions are factual and grounded in historical financial records.

## Citations

```bibtex
@article{opsahlong2025officeqapro,
      title={OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning},
      author={Krista Opsahl-Ong and Arnav Singhvi and Jasmine Collins and Ivan Zhou and Cindy Wang and Ashutosh Baheti and Owen Oertell and Jacob Portes and Sam Havens and Erich Elsen and Michael Bendersky and Matei Zaharia and Xing Chen},
      year={2026},
      eprint={2603.08655},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.08655},
}
```
