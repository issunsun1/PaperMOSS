# LLM API Notes

## Supported Mode

- Summary provider: `openai-compatible`
- Endpoint style: OpenAI Chat Completions API

## Required Inputs

- `--llm-api-base`
- `--llm-model`
- Optional key input: `--llm-api-key`
- Environment fallback: `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`

## URL Resolution

- If base ends with `/v1`, request `/v1/chat/completions`.
- If base already ends with `/chat/completions`, use it directly.
- Otherwise append `/v1/chat/completions`.

## Performance Controls

- `--llm-workers` controls summarization concurrency.
- `--llm-max-papers` limits how many papers are sent to LLM.
- `--summary-max-chars` truncates long LLM outputs.

## Failure Behavior

- Any LLM request failure falls back to extractive summary for that paper.
- Warning messages are included in output for traceability.
