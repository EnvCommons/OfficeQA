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

- **Dataset** (question CSVs): [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Code** (reward.py): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Treasury Bulletin documents**: U.S. Government public domain

## Tasks

| Split | Source CSV | Count | Description |
|-------|-----------|-------|-------------|
| `train` | `officeqa_full.csv` | ~246 | Full development set (easy + hard) |
| `test` | `officeqa_pro.csv` | ~133 | Benchmark set (hard only) |

Each task contains a natural language question, difficulty rating, and pointers to relevant source documents.

## Reward Structure

Sparse, verifiable, binary reward (1.0 or 0.0). Scoring uses deterministic fuzzy numerical matching:

- Unit-aware comparison (million, billion, trillion, thousand)
- Configurable tolerance (default: exact match)
- Multi-number list support
- Text overlap checking for hybrid answers (e.g., "March 1977")
- `<FINAL_ANSWER>` tag extraction

## Data

- **Corpus**: 696 transformed Treasury Bulletin text files (~200 MB) with Markdown-formatted tables
- **Source**: [databricks/officeqa](https://github.com/databricks/officeqa) on GitHub
- **Mount path**: `/home/ubuntu/documents/` in sandbox

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands — search documents, run Python, access the web |

Grading uses a hidden `@terminal` tool: when the agent is done, it replies with a plain message wrapping the final answer in `<FINAL_ANSWER>...</FINAL_ANSWER>` tags. The environment extracts the tagged answer (falling back to the whole reply if no tags) and scores it with fuzzy numeric/text matching against the reference.

## Time Horizon

Multi-turn. Agents typically perform 10–50+ tool calls: searching documents, reading files, writing and executing Python scripts, and optionally querying external data sources before finishing with a plain-text reply.

## Environment Difficulty

Frontier models achieve ~57% accuracy on the Pro (hard) set. Human performance is ~34–51% depending on setup. 62% of questions require analytical reasoning beyond basic arithmetic.

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
