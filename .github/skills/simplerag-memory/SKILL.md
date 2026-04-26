---
name: simplerag-memory
description: 'Use for remembering, recalling, logging, noting, storing, retrieving, searching, or forgetting ad hoc information with SimpleRAG. Trigger on phrases like remember this, log this, note this, what did I tell you, recall, retrieve, show my notes, and forget this.'
user-invocable: true
disable-model-invocation: false
---

# SimpleRAG Memory

Use this skill when the user wants to save short-to-medium notes, retrieve prior notes semantically, inspect available memory groups, or delete previously stored memory items.

## When to Use
- Remember or log information for later retrieval.
- Recall what the user said previously about a topic.
- Retrieve notes by semantic meaning instead of exact keywords.
- Show available note groups.
- Forget a note when a concrete `doc_id` is known.

## Backend
This skill uses a hosted SimpleRAG API.

Primary endpoint:
- `http://192.168.1.70:7000`

Fallback endpoint:
- `http://mghasemi.ddns.net:7000`

If the primary endpoint is unavailable, retry against the fallback endpoint before reporting an error.

## Prerequisites
- `curl` must be available for HTTP requests.
- `jq` must be available for JSON-safe payload construction in the helper script.

## Routing
- Store intent: `remember`, `log`, `note`, `save this`, `store this` -> store
- Recall intent: `recall`, `what did I tell you`, `retrieve`, `search my notes` -> query
- List intent: `show my notes`, `list groups`, `what memory groups exist` -> groups
- Forget intent: `forget`, `remove that note`, `delete memory` -> delete

## Defaults
- Default `group_id` to `workspace` unless the user specifies one.
- Let the API auto-generate `doc_id` unless the user gives one.
- Use `k=5` by default for recall.
- Only include `ingested_after` when the user asks for recent or time-bounded recall.

## Safety Rule
If the user asks to forget a note but no `doc_id` is known, first help locate the note by querying or listing groups. Do not guess a deletion target.

## Procedure
1. Determine whether the user wants to store, recall, list, or forget.
2. Use the hosted SimpleRAG helper script to call the API.
3. Return concise results with the important identifiers.
4. For store, include the assigned `doc_id`.
5. For recall, return the top 1-3 results with score and ingestion date.
6. For delete, confirm the deleted `doc_id` and chunk count.

## Worked Examples
1. `Remember that we use cosine distance for retrieval.`
	Store in `workspace` and respond with the assigned `doc_id` and chunk count.
2. `What did I tell you about cosine distance?`
	Query `workspace` and return the top 1-3 matching snippets with score.
3. `Show my notes.`
	Call groups and return group names with document counts.
4. `Forget doc auth-2026-04-23.`
	Delete that exact `doc_id` and confirm the deleted chunk count.
5. `What recent notes do I have about embeddings?`
	Query with `ingested_after` only if the user supplied or implied a time bound.

## Helper
Use [the SimpleRAG client](./scripts/simplerag_client.sh) for API calls.

## Reference
See [API usage and conventions](./references/api-usage.md) for payloads, endpoint failover, and response formatting.
See [example prompts](./references/examples.md) for concrete remember/recall/list/forget utterances.
