> **NOTE:** This is the template for your report. Delete this note block before submission.

# Align a Small LLM with GRPO for Strict JSON Generation
- **Group ID**: [E.g., G07]
- **Project ID**: [E.g., 1]

---

## 1. Introduction and Objective
*Describe the objective of the project, why it is relevant, and what specific problem you are trying to solve. What is your main goal or initial hypothesis?*

## 2. Contribution and Added Value
*Summarize what was done concisely. ("We built a model based on X for task Y"). Highlight your added value compared to merely running existing code (e.g., new loss, different architecture, specific data augmentation, etc.).*

## 3. Data Used
*Describe formally the used data:
- Where do they come from?
- What are the main statistics (number of samples, train/test split)?
- What kind of preprocessing or augmentation did you apply to prepare them for training?*

## 4. Methodology and Architecture

### 4.1 Overview

*Detail the experiments and how your system is built. What architecture did you use as a baseline? How did you modify it? Describe the network topology, key layers, the loss function used, and the training logic.*

### 4.2 Reward Functions

GRPO replaces the traditional reward model with rule-based reward functions that provide a dense, interpretable signal. Each completion is scored independently by five components; GRPOTrainer computes the weighted sum $r = \sum_{i} w_i \cdot r_i$ and normalises across the generation group to obtain per-sample advantages.

#### Component summary

| Component | Range | Weight | Purpose |
|:---|:---:|:---:|:---|
| `format_reward` | $[0, 1]$ | 0.25 | Checks for a proper ` ```json ... ``` ` code fence (1.0), a generic ` ``` ... ``` ` fence (0.5), or no fence (0.0). |
| `validity_reward` | $[0, 1]$ | 0.30 | Graduated score based on JSON parseability. Valid JSON → 1.0; for invalid JSON the score is proportional to how far into the string the first parse error occurs (0.70 if error in the last 15%, down to 0.20 if in the first 40%). |
| `schema_reward` | $[0, 1]$ | 0.30 | Average of constraint checks extracted from the instruction: exact/minimum array count, required key presence, nesting depth, top-level type (array vs object). Returns 1.0 if JSON is valid but no constraints are extractable. |
| `truncation_reward` | $[-1, +1]$ | 0.15 | Detects completions truncated mid-generation (hitting `max_completion_length`). Returns 1.0 for structurally complete output, 0.0 if no JSON is detected (neutral), and **−1.0** for bare JSON with unclosed braces/brackets, trailing commas, or unterminated strings. |
| `reasoning_reward` | $[0, 1]$ | 0.0 | Bonus for `<think>…</think>` reasoning blocks (≥ 20 chars). Disabled (`weight = 0`) in the current configuration (`thinking: false`). |

#### Design rationale

- **Additive, not gating.** Each component contributes independently to the total reward. This avoids the "zero advantage" problem: if all completions in a GRPO group scored exactly 0 (e.g., because of a hard gate on format), the advantage standard deviation would be 0 and the policy gradient would vanish — the model would never learn to produce code blocks.

- **No length bias on correct output.** Both `validity_reward` and `schema_reward` return 1.0 for any valid, constraint-satisfying JSON regardless of length. The graduated scores in `validity_reward` (0.20–0.70) apply only to *invalid* JSON and reflect how much of the string was correct before the error — this is intentional partial credit to provide gradient signal during early training. `schema_reward` checks structural properties (count, keys, depth, type), not string length.

- **Truncation as negative reward.** The `truncation_reward` is the only component that returns negative values. Its effective influence is $0.15 \times 2.0 = 0.30$ (weight × range amplitude), matching the influence of `validity_reward` and `schema_reward` ($0.30 \times 1.0 = 0.30$ each). This ensures truncated completions receive a strong penalty without dominating the overall signal.

- **Weight redistribution.** When `thinking: false`, the `reasoning_reward` weight is redistributed proportionally across all remaining active components, preserving their relative ratios (e.g., format 0.25 : validity 0.30 : schema 0.30 : truncation 0.15 stays in the same ratio).

### 4.3 Curriculum Learning

*[TODO: describe the 3-stage curriculum approach — format_basics, progressive, full_difficulty — with difficulty weight schedules and per-stage temperature]*

## 5. Results and Discussion
*Insert here the quantitative tables with the achieved results and compare your solution with the baseline. **Do not limit yourself to pasting numbers**, but comment on them:
- Why does model A perform better than model B?
- Are there classes where the model is particularly weak?
- Show qualitative examples (e.g., inserting correctly vs. incorrectly predicted images).*

Example:

**Table 1**: Quantitative results

| Model | Metric 1 | Metric 2 |
| :--- | :---: | :---: |
| Baseline | 65.4% | 45.1 |
| Our Final Model | 72.1% | 50.4 |

## 6. Conclusion and Limitations
*Summarize the project's outcome. What are the current limitations (e.g., requires too much memory, fails in low-light conditions)? If you had more time, what future experiments would you run?*

## 7. Additional Information

### 7.1 Contribution Breakdown
*Detail clearly who did what within the group.*
- **Person 1**: ...
- **Person 2**: ...
- **Person 3**: ...

### 7.2 Use of Artificial Intelligence
*Declare here the possible use of tools like Copilot or ChatGPT, specifying in which phases they helped you (e.g., writing boilerplate, debugging, documentation), keeping in mind that the architectural design and the responsibility for the result are yours.*
