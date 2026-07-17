# llm-alpha-generator

Formulaic alpha-factor mining for stocks and futures. The current tool-host LLM is used by default; a complete external model configuration may be supplied only through `config.llm`.

All LLM stages are mandatory: candidate generation, dimension inference, AlphaEval logic scoring, and economic explanation. Failures are fatal and no report is published.

See `SKILL.md` for the workflow and `python scripts/test.py` for validation.
