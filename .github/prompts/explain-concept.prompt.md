---
description: Explain a technical concept from the project in plain English
mode: ask
---

Explain the requested technical concept (acronym, method, or finding) for a non-technical audience.

## Format

Three sections:

1. **What it actually is** — the technical definition, briefly
2. **Why it matters for this project** — the role it plays in the ML-PPA framework
3. **Plain-English analogy** — a comparison from everyday experience

## Style

- **Use concrete numbers from the project** when relevant (-17.6, $8.35, 7/7, etc.)
- **Avoid jargon chains.** If you use a technical term to explain another technical term, define both.
- **Don't be condescending.** The audience is intelligent, just not chip-design-specialized.
- **Keep it tight.** Three short paragraphs is usually enough.

## Reference: terms commonly asked about

When explaining, use the canonical framings from the paper:

- **EDA:** Software for designing chips. ORFS is the open-source one we use.
- **RL:** Trial-and-error machine learning. The agent picks clock targets, gets a score, learns over 500K iterations.
- **BO (Bayesian Optimization):** Smart search that uses uncertainty estimates to decide where to look next.
- **GAT:** A neural network that operates on graphs. We use it to predict chip quality from netlist topology.
- **GBDT:** A simpler tree-based model. Fast and accurate when you have tabular features. We use it for RL feedback.
- **PPA:** Power, Performance, Area — the three things every chip designer trades off.
- **Pareto front:** The line of trade-offs you literally can't beat without giving something up. CAND-6 went past CTRL-2's position on this line.
- **Surrogate model:** A fast approximation of a slow process. Our GBDT and GAT are surrogates for full EDA flows.
- **WNS / timing slack:** How much margin you have on the chip's clock speed. Negative WNS means timing failed.
- **MAE / prediction error:** Average distance between predicted and actual values. Lower is better.
- **IQM:** Interquartile mean — a robust average that ignores outliers. We use it for RL evaluation.

## Bad explanations to avoid

- Don't say "it's like AI but for chips" — too vague
- Don't say "the algorithm finds the best design" — overselling, and "best" is wrong scope
- Don't compare to specific commercial products unless accurate
- Don't claim this would "replace human chip designers" — not what the project shows

## Good explanation pattern

> **What it actually is:** [Technical definition in 1-2 sentences]
>
> **Why it matters for this project:** [Specific role, with project numbers]
>
> **Plain-English analogy:** [Concrete everyday comparison]

If the concept is genuinely complex (not just an acronym), it's okay to use 4 paragraphs instead of 3 — but no more.
