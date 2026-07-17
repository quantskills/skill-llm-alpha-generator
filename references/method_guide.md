# Method Guide

## Workflow

1. Freeze and validate OHLCV data.
2. Build registered features and require LLM dimension/semantic votes.
3. Generate diverse candidate formulas with `emit_alpha_candidates`.
4. Apply whitelist, dimension, and look-ahead checks.
5. Warm-start genetic programming using only the training slice.
6. Score selected factors with all AlphaEval dimensions, including mandatory LLM logic scoring.
7. Generate structured economic explanations with `emit_explanation`.
8. Publish the HTML report atomically only after every required LLM stage succeeds.

## Boundaries

- This skill mines and evaluates factors; it does not perform a formal backtest.
- Numeric scores come from deterministic evaluation except the logic dimension, which must come from LLM.
- Explanations must describe mechanism, applicable conditions, and failure conditions; formula paraphrase alone is invalid.
- Every iteration must retain its best formula, fitness, mean fitness, diversity, and candidate snapshot.

See `references/llm_policy.md` for the mandatory runtime contract and `references/output_contract.md` for result fields.
