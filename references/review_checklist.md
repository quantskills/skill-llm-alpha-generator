# Review Checklist

- [ ] `SKILL.md` and `agents/` metadata are present.
- [ ] Only the single `config.llm` entry is used for external LLM configuration.
- [ ] Candidate generation, dimension inference, AlphaEval logic, and explanation all call the shared runtime.
- [ ] No environment switch, private client, neutral fallback, template explanation, or pure-GP downgrade exists.
- [ ] `meta.llm_runtime` includes run/request identifiers and payload hashes.
- [ ] Iteration trace includes one record per GP generation.
- [ ] Report publication is atomic and never occurs after a failed stage.
- [ ] `python scripts/test.py` passes.
