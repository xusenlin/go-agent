# Thinking Level Demo

This example demonstrates the thinking level feature in the go-agent framework.

## Overview

The thinking level feature allows you to control the depth of reasoning the LLM uses when processing your requests. This can be useful for:

1. **Cost control**: Lower thinking levels use fewer reasoning tokens
2. **Performance tuning**: Adjust the depth of reasoning for different types of tasks
3. **Quality control**: Ensure the model has enough reasoning depth for complex tasks

## Usage

```go
agent.New().
    WithProvider(anthropicprovider.New(apiKey, nil)).
    WithModel("claude-opus-4-7").
    WithThinkingLevel(provider.ThinkingLevelHigh). // thinking/reasoning depth level
    Build()
```

## Parameters

- `WithThinkingLevel(level provider.ThinkingLevel)`: Sets the thinking/reasoning depth level
  - `provider.ThinkingLevelNone` (default): No thinking
  - `provider.ThinkingLevelLow`: Low thinking depth
  - `provider.ThinkingLevelMedium`: Medium thinking depth  
  - `provider.ThinkingLevelHigh`: High thinking depth

## Provider Support

### Anthropic
The thinking level is mapped to token budgets:
- Low: 2000 tokens
- Medium: 5000 tokens
- High: 10000 tokens

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  }
}
```

### Gemini
The thinking level enables thoughts in the response:
```go
cfg.ThinkingConfig = &genai.ThinkingConfig{
    IncludeThoughts: true,
}
```

### OpenAI
The thinking level is passed as `reasoning_effort`:
```json
{
  "reasoning_effort": {
    "effort": "high"
  }
}
```

## Example

```bash
ANTHROPIC_API_KEY=sk-... go run ./examples/thinking-level
```

## Notes

- The thinking level is set at agent creation time and applies to all requests made by that agent
- Different providers may have different implementations for thinking/reasoning
- The actual token usage may vary depending on the provider and model