# Section-by-Section Writing Guide for CS Research Papers

This reference provides detailed conventions for each major section of a computer science research paper. Use it when refining a specific section to ensure the output matches what reviewers at top venues expect.

## Table of Contents
1. [Title](#title)
2. [Abstract](#abstract)
3. [Introduction](#introduction)
4. [Related Work](#related-work)
5. [Methodology / Approach](#methodology)
6. [Experiments](#experiments)
7. [Results and Analysis](#results-and-analysis)
8. [Discussion](#discussion)
9. [Conclusion](#conclusion)
10. [Common Cross-Section Issues](#common-cross-section-issues)

---

## Title

A good title is specific, informative, and concise (typically 8–15 words).

**Patterns that work**:
- "[Method Name]: [What It Does] for [Problem Domain]"
- "[Verb]-ing [Problem] via [Approach]"
- "[Descriptive Phrase] for [Task]"

**Avoid**:
- Titles that are just a method name with no indication of what it does
- Questions as titles (unless the paper genuinely investigates a question and the venue accepts this style)
- Excessive punctuation, colons, or nested subtitles
- Clickbait or hype: "Revolutionary", "Game-Changing", "Towards Ultimate"

**Check**: Can a researcher scanning a proceedings page understand what this paper is about from the title alone?

---

## Abstract

Target: 150–250 words (check venue limits). Must be entirely self-contained.

**Structure** (roughly one sentence each, expand as needed):
1. **Context/Problem**: What problem exists and why it matters
2. **Gap**: What current approaches fail to address
3. **Approach**: What this paper proposes (name the method)
4. **Key insight**: What makes the approach work (the "why")
5. **Results**: Concrete numbers on primary benchmarks
6. **Significance**: Why these results matter

**Rules**:
- No citations (the abstract should stand alone)
- No undefined acronyms — spell out on first use or avoid
- Include at least one concrete quantitative result
- Do not start with "In this paper, we..." — start with the problem or context
- Avoid: "In recent years", "With the rapid development of", "has attracted growing attention"

**Quality test**: If a reader reads only the abstract, do they know (1) the problem, (2) the approach, (3) the main result?

---

## Introduction

Typically 1–1.5 pages. The most read section after the abstract.

**Structure**:
1. **Opening paragraph**: Establish the problem domain and its importance. Ground it in something concrete — a real-world need, a fundamental limitation, a compelling example. Avoid starting with truisms ("Deep learning has achieved remarkable success...").
2. **Problem specifics**: Narrow from the broad domain to the specific problem this paper addresses. What makes it challenging?
3. **Limitations of existing work**: What have others tried and where do they fall short? This should motivate your approach without being a full related work section. Be fair — characterize prior work accurately.
4. **Your approach**: Introduce your method at a high level. What is the key idea? Why should it work where others failed?
5. **Contributions**: Explicitly list 2–4 contributions using either a bulleted list or inline enumeration. Contributions should be specific and verifiable ("We propose X that achieves Y" not "We study the problem of Z").
6. **Results preview** (optional): Brief mention of headline results to build confidence.
7. **Paper outline** (optional, venue-dependent): "The remainder of this paper is organized as follows..." — some venues expect this, others find it wasteful. Include if the paper structure is non-standard.

**Common pitfalls**:
- Overclaiming: "We are the first to..." — be careful. "To the best of our knowledge" helps, but verify.
- Underclaiming: Burying the contribution in vague language. Be direct about what you did.
- Motivating a solution instead of a problem: Don't start by saying "We propose X". Start by saying why X is needed.

---

## Related Work

Typically 0.75–1.5 pages. Position your work within the landscape.

**Organize by theme, not by paper**. Group related papers under subheadings:
- "Graph Neural Networks for Molecular Property Prediction"
- "Uncertainty Quantification in Language Models"
- "Active Learning for Structured Prediction"

**Each paragraph should**:
1. Describe what this line of work does (collectively, not paper by paper)
2. Highlight key approaches and findings
3. Contrast with the current paper — what gap remains?

**Avoid "laundry list" style**:
- Bad: "Smith et al. (2020) proposed X. Jones et al. (2021) extended this to Y. Lee et al. (2022) further improved upon Y by using Z."
- Better: "Several approaches have addressed X by building on the framework of Smith et al. (2020). Jones et al. (2021) extended this to handle Y, while Lee et al. (2022) improved scalability through Z. However, these methods share a common limitation: they assume..."

**End each thematic group** by distinguishing your work: "In contrast to these approaches, our method..."

**Be generous and fair**: Cite relevant work thoroughly. Reviewers are often authors of papers you should be citing.

---

## Methodology

The core technical section. Length varies (2–4 pages typical).

**Structure recommendations**:
1. **Overview**: A high-level description (1 paragraph) and optionally a figure showing the architecture or pipeline
2. **Preliminaries/Problem Formulation**: Define the problem formally. Introduce notation. State assumptions.
3. **Method details**: Present in logical order — each component should build on what came before
4. **Key design choices**: Explain why you made the choices you did, not just what they are

**Writing principles**:
- **Define before use**: Every symbol, every term, every abbreviation — define it before or at first use
- **Equations need prose**: Every equation should be preceded by motivation ("To capture the interaction between X and Y, we define:") and followed by interpretation ("Intuitively, this measures...")
- **Number your equations** if you refer to them later. Do not number equations you never reference.
- **Consistent notation**: Pick a convention and stick with it. Lowercase bold for vectors ($\mathbf{x}$), uppercase bold for matrices ($\mathbf{W}$), calligraphic for sets ($\mathcal{D}$), etc.
- **Avoid notation overload**: If you have more than 15–20 symbols, consider a notation table

**Common issues**:
- Jumping into equations without motivation
- Defining notation in an equation environment (put variable definitions in text)
- Inconsistent subscript/superscript conventions
- Missing dimensionality information (what size is $\mathbf{W}$?)

---

## Experiments

Typically 2–3 pages. This is where you prove your claims.

**Structure**:
1. **Research questions or hypotheses** (optional but strong): "We design experiments to answer: (RQ1) Does X improve over Y? (RQ2) How does Z affect performance?"
2. **Datasets**: Name, size, domain, train/dev/test splits, preprocessing, why these datasets
3. **Baselines**: What you compare against, why these baselines, ensure they are fair comparisons
4. **Implementation details**: Hyperparameters, training procedure, hardware, runtime. Enough for reproducibility.
5. **Evaluation metrics**: What you measure and why

**Writing tips**:
- **Baselines should be strong and recent**. Reviewers will notice if you only compare against outdated methods.
- **Be explicit about what is fair**: Same data splits? Same pretraining? Same compute budget?
- **Hyperparameter reporting**: State how hyperparameters were selected (grid search, validation set, etc.)
- **Reproducibility**: Include random seeds, number of runs, variance/standard deviation where applicable

---

## Results and Analysis

Can be combined with Experiments or separate. This is where numbers meet narrative.

**Presenting results**:
- Lead with the main result table/figure, then walk the reader through it
- Highlight the most important comparisons — do not just list all numbers
- Report statistical significance or confidence intervals when possible
- Bold the best result in tables. Use underline or second-best marking if venue convention supports it.

**Analysis should**:
- **Explain why**, not just what: "Our method outperforms X by 3.2 points, which we attribute to the ability of component Y to capture long-range dependencies"
- **Include ablation studies**: What happens when you remove each component?
- **Show failure cases**: Where does your method struggle? This builds credibility.
- **Error analysis**: Especially valued at NLP venues (ACL, EMNLP). Categorize errors and explain patterns.

**Avoid**:
- Cherry-picking results — report performance on all standard metrics, even where you are not best
- Overclaiming marginal improvements — if the difference is within noise, say so
- Tables without discussion — never present a table and move on

---

## Discussion

Optional section (some papers fold this into Results or Conclusion).

**Include when**:
- The results raise interesting questions that deserve exploration
- There are limitations that need honest acknowledgment
- The work has broader implications worth discussing

**Limitations subsection**: Increasingly expected (NeurIPS requires it). Be specific and honest. "Our method assumes X, which may not hold when Y." This is a strength, not a weakness — it shows intellectual honesty and helps future researchers.

---

## Conclusion

Typically 0.5–0.75 pages. Do not repeat the abstract.

**Structure**:
1. One-sentence restatement of what you did and why
2. Key contributions (brief — the reader has seen the details)
3. Main takeaway or insight
4. Limitations (if no separate discussion section)
5. Future work — be specific. "Extending to other domains" is vague. "Applying our calibration method to multi-turn dialogue systems where confidence estimates are particularly critical" is concrete.

**Avoid**:
- Restating the full methodology
- Introducing new information
- Excessive hedging or false modesty
- Grandiose claims about impact

---

## Common Cross-Section Issues

**Tense consistency**:
- Use present tense for general truths and descriptions of your method: "Our model uses attention..."
- Use past tense for experimental actions: "We trained the model for 50 epochs"
- Use present tense for results in tables: "Table 2 shows that..."

**Citation style**:
- "Smith et al. (2023) showed..." (narrative citation — the authors are the subject)
- "This has been shown previously (Smith et al., 2023)" (parenthetical citation — the work supports a claim)
- Do not use "In [23]" — use author names for readability

**Figures and tables**:
- Every figure and table must be referenced in the text
- Captions should be self-contained — a reader should understand the figure without reading the main text
- Place figures/tables near their first reference
- Use consistent formatting across all tables

**Numbers and units**:
- Use consistent decimal places (if one baseline has 85.3, don't report yours as 87.34)
- Include units where applicable
- Use thousands separators for large numbers: 1,000,000 not 1000000

**Acronyms**:
- Define on first use in both abstract and body (they are separate contexts)
- Do not define acronyms you use only once — just spell it out
- Common venue-specific acronyms (NLP, LLM, GNN) may not need definition depending on venue
