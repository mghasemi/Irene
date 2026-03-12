# Rebuttal and Author Response Guide

This reference covers writing effective rebuttals and author responses for top CS conference peer review. Use when a user is responding to reviewer comments.

## Principles

A rebuttal is a professional conversation, not a defense. The goal is to address concerns clearly, provide missing information, and demonstrate that the paper's contributions are sound. Reviewers are volunteers who gave time to read your work — treat their feedback with respect, even when you disagree.

## Structure

### Opening

Start with a brief thank-you (one sentence) and a summary of changes or clarifications you will provide. Do not be obsequious.

Example: "We thank the reviewers for their thoughtful feedback. Below we address each concern."

### Per-Reviewer Responses

Address each reviewer separately, quoting their concern and responding directly.

**Format**:
```
**Reviewer [X], Comment [N]**: [Brief quote or paraphrase of concern]

**Response**: [Your response]
```

### Response Types

**For factual corrections** (reviewer misunderstood something):
- Point to the specific location in the paper
- Quote the relevant text
- Explain what it means
- Offer to clarify the wording: "We will revise Section X to make this clearer"

**For missing experiments/analysis**:
- If you can run the experiment: provide the results directly in the rebuttal
- If you cannot: explain why (time/resource constraints) and commit to adding it in the revision
- Never promise what you cannot deliver

**For conceptual disagreements**:
- Acknowledge the reviewer's perspective
- Present your reasoning clearly with evidence
- Cite relevant literature if helpful
- Be respectful — "We appreciate this perspective and would like to offer an alternative view" not "The reviewer is incorrect"

**For limitations/weaknesses acknowledged**:
- Agree when the reviewer is right
- Explain what you plan to do about it
- If it is out of scope, explain why while acknowledging the point

## Tone Guide

**Do**:
- Be direct and specific
- Provide evidence (numbers, citations, quotes from the paper)
- Acknowledge valid points
- Commit to concrete revisions
- Keep responses concise — word limits are tight

**Do not**:
- Be defensive or dismissive
- Say "the reviewer misunderstood" — instead, say "we will clarify"
- Make vague promises: "we will improve the paper"
- Ignore difficult questions — address everything
- Repeat large blocks of the paper — summarize and reference

## Word Count Management

Most venues have strict word or page limits for rebuttals. Prioritize:
1. Major concerns that could affect the accept/reject decision
2. Factual misunderstandings that change the assessment
3. Requests for additional experiments you can address
4. Minor points (address briefly or batch together)

## Common Reviewer Concerns and Response Patterns

**"The contribution is incremental"**:
Highlight what is novel. Provide quantitative evidence of improvement. Explain the practical or theoretical significance. Do not just restate the contributions — add context the reviewer may have missed.

**"Missing comparison to [method X]"**:
If you can add it: "We ran this comparison. [Method X] achieves [score] on [dataset], while ours achieves [score]."
If you cannot: Explain why the comparison is not straightforward (different setting, code unavailable, etc.) and cite any available related results.

**"The writing needs improvement"**:
Acknowledge and commit: "We will carefully revise the paper for clarity. Specifically, we will [concrete changes]."

**"Limited evaluation"**:
Add new results if possible. If not, explain the rationale for your current evaluation choices and commit to expanding in the revision.

**"The assumptions are too strong"**:
Discuss when the assumptions hold in practice. If possible, show empirical evidence that the method works even when assumptions are partially violated. Acknowledge the limitation and discuss relaxation as future work.
