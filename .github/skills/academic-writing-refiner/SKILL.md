---
name: academic-writing-refiner
description: Refine academic writing for computer science research papers targeting top-tier venues (NeurIPS, ICLR, ICML, AAAI, IJCAI, ACL, EMNLP, NAACL, CVPR, WWW, KDD, SIGIR, CIKM, and similar). Use this skill whenever a user asks to improve, polish, refine, edit, or proofread academic or research writing — including paper drafts, abstracts, introductions, related work sections, methodology descriptions, experiment write-ups, or conclusion sections. Also trigger when users paste LaTeX content and ask for writing help, mention "camera-ready", "rebuttal", "paper revision", or reference any academic venue or conference. This skill handles both full paper refinement and section-by-section editing.
---

# Academic Writing Refiner

This skill transforms rough or intermediate academic drafts into polished, publication-ready prose for top-tier CS conferences. The goal is writing that is clear, precise, and accessible to a broad technical audience — the kind of writing that reviewers at venues like NeurIPS, ICML, or ACL appreciate because it respects their time and communicates ideas efficiently.

## Core Philosophy

Top CS conferences share a common expectation: writing should be a transparent window into the ideas, not a display of vocabulary. The best papers at NeurIPS, ACL, or KDD succeed not because they use impressive words, but because every sentence earns its place and every paragraph advances the reader's understanding.

This means:
- **Clarity over cleverness**: Use the simplest word that precisely conveys the meaning. "Use" instead of "utilize", "show" instead of "demonstrate" (unless you mean a formal proof/demonstration), "many" instead of "a plethora of".
- **Precision over vagueness**: Replace hedging language with specific claims. Instead of "our method performs quite well", say "our method achieves 94.3% accuracy, outperforming the strongest baseline by 2.1 points".
- **Economy over verbosity**: Every sentence should do work. If removing a sentence doesn't lose information, remove it.
- **Flow over fragmentation**: Guide the reader from one idea to the next with logical connectives, not abrupt jumps.

## How to Refine

When a user provides text to refine, follow this process:

### 1. Understand the Context

Before editing, figure out:
- **What section is this?** (abstract, introduction, related work, methodology, experiments, conclusion) — each has different conventions.
- **What venue?** If stated, tailor to that venue's style norms. ML venues (NeurIPS, ICML, ICLR) tend toward concise, equation-heavy writing. NLP venues (ACL, EMNLP, NAACL) often expect more linguistic precision and thorough related work. IR/Web venues (SIGIR, WWW, KDD, CIKM) often need clear problem motivation tied to practical impact.
- **What stage?** A first draft needs structural help; a camera-ready needs polish.

If the user doesn't specify, infer from content and ask only if genuinely ambiguous.

### 2. Apply Section-Specific Conventions

Read `references/section-guide.md` for detailed conventions per section type. The key principles:

**Abstract**: Should be self-contained, state the problem, approach, key result (with numbers), and significance — all in ~150–250 words. No citations, no undefined acronyms.

**Introduction**: Problem → gap → contribution → brief results → paper outline. The reader should understand what you did and why it matters within the first page.

**Related Work**: Group by theme, not by paper. Each paragraph should end by distinguishing the current work from what was just discussed. Avoid "laundry list" style (X did A. Y did B. Z did C.).

**Methodology**: Present the approach in logical order. Define notation before using it. Use equations for precision but always provide intuition in words alongside them.

**Experiments**: Lead with research questions or hypotheses, then describe setup, then results. Tables and figures should be self-contained with descriptive captions.

**Conclusion**: Summarize contributions (not the whole paper), acknowledge limitations honestly, suggest concrete future directions.

### 3. Sentence-Level Refinement

Consult `references/word-choice.md` for a quick-reference table of common substitutions (fancy → simple, filler → delete, hedging calibration, and transition connectives). Apply these transformations systematically:

**Tighten prose**:
- Remove filler phrases: "it is worth noting that", "it should be mentioned that", "in order to" → "to"
- Eliminate redundancy: "completely eliminate" → "eliminate", "future plans" → "plans"
- Convert passive to active where it improves clarity: "the model was trained by us" → "we trained the model"
- But keep passive voice when the agent is unimportant: "the dataset was collected from public sources" is fine

**Fix common academic writing issues**:
- Dangling modifiers: "Using gradient descent, the loss decreases" → "Using gradient descent, we minimize the loss"
- Noun pile-ups: "multi-task learning based pre-trained language model fine-tuning approach" → break it up with prepositions
- Vague referents: "This shows that..." — what does "this" refer to? Make it explicit
- Orphan claims: every claim about performance needs a citation or experimental reference

**Strengthen transitions**:
- Between sentences: use logical connectives that signal the relationship (however, therefore, specifically, in contrast, building on this)
- Between paragraphs: the first sentence of each paragraph should connect to the previous paragraph's conclusion
- Between sections: the last paragraph of a section should preview what comes next

### 4. LaTeX-Specific Handling

When the input contains LaTeX:
- Preserve all `\cite{}`, `\ref{}`, `\label{}`, equation environments, and custom macros exactly as written
- Fix only the prose — do not modify mathematical content unless there is a clear notational inconsistency
- Maintain `\textbf{}`, `\textit{}`, `\emph{}` formatting choices
- Ensure consistent notation: if the user writes $\mathbf{x}$ in one place and $\boldsymbol{x}$ in another for the same quantity, flag it
- Keep `~` (non-breaking spaces) before `\cite` and `\ref`
- Preserve `%` comments
- Do not add or remove `\paragraph{}`, `\subsubsection{}` etc. unless the user asks for structural changes

### 5. What NOT to Do

These are equally important as what to do:
- **Do not insert fancy vocabulary**. "Leverage" is almost never better than "use". "Elucidate" is almost never better than "explain". If the original uses a simple word correctly, keep it.
- **Do not over-hedge**. Academic writing needs appropriate qualification ("may", "suggests"), but excessive hedging ("it could potentially be argued that this might possibly indicate") undermines confidence in the work.
- **Do not add content**. Refine what is there. If something is missing (e.g., no related work comparison, no baseline), flag it as a suggestion but do not invent claims or results.
- **Do not homogenize voice**. If the author has a distinct (but correct) style, preserve it. The goal is to polish, not to flatten.
- **Do not use em-dashes excessively**. Parentheses or restructured sentences are usually cleaner in academic writing. One em-dash pair per paragraph at most.
- **Do not introduce semicolons liberally**. Prefer shorter sentences joined by appropriate connectives over long semicolon-connected chains.

## Output Format

When presenting refined text:

1. **Provide the refined version** as the primary output, clearly separated from commentary
2. **Add brief marginal notes** for substantive changes — explain why you changed something when the reason isn't obvious (e.g., "Restructured to lead with the contribution rather than the gap" or "Made the comparison to X explicit")
3. **Flag issues you cannot fix** — missing citations, unclear experimental details, potential factual concerns — as a separate list at the end
4. If the input is LaTeX, output LaTeX. If the input is plain text, output plain text. Match the format.

## Interaction Patterns

**Full paper refinement**: If the user provides an entire paper (or most of one), work section by section. Start with whichever section the user indicates, or begin with the abstract and introduction since those set the tone.

**Single section**: Apply the full refinement process to that section.

**Quick polish**: If the user says "just fix the grammar" or "light edit only", respect that — fix spelling, grammar, and punctuation without restructuring or rewriting.

**Iterative refinement**: After providing a refined version, be ready for feedback like "too formal", "I want to keep the original structure of paragraph 2", or "make the motivation stronger". Apply changes surgically without re-editing the rest.

**Rebuttal writing**: When the user mentions a rebuttal or reviewer response, read `references/rebuttal-guide.md` for specific advice on crafting effective rebuttals.

## Common Venue-Specific Notes

| Venue Group | Style Tendencies |
|---|---|
| NeurIPS, ICML, ICLR | Concise, equation-centric. Theoretical rigor valued. Anonymous review — remove self-identifying references. |
| AAAI, IJCAI | Broader AI scope. Motivation and real-world relevance important. Slightly more expository than ML-focused venues. |
| ACL, EMNLP, NAACL | Thorough related work expected. Linguistic precision in terminology. Error analysis and ablation studies valued. |
| CVPR | Visual results critical. Qualitative examples alongside quantitative. Clear figure descriptions. |
| WWW, KDD, SIGIR, CIKM | Problem-driven motivation. Scalability and practical impact often expected. Dataset descriptions need care. |

These are tendencies, not rigid rules — good writing is good writing regardless of venue.
