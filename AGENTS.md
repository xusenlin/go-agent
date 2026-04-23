# Agent Documentation

## Project Overview

Go-based ReAct (Reason + Act) agent framework with multi-provider LLM support, streaming, composable hooks, and MCP integration.

## Build & Run Commands

```bash
# Build
go build ./...

# Run examples (use go run to avoid creating binaries)
go run ./examples/plan
go run ./examples/thinking-level
go run ./examples/stream

# Run tests
go test ./...

# Dependency management
go mod tidy
```

**Note**: Use `go run` instead of `go build` when testing examples to avoid creating binary files in the project directory.

## Configuration

The project uses a `.env` file for configuration. Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

The `.env` file contains:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: The model to use (default: gpt-4o)
- `OPENAI_BASE_URL`: The API base URL (default: https://api.openai.com/v1)

All examples and tests automatically load configuration from the `.env` file and use the OpenAI protocol.

## Architecture

```
agent.New() (Builder)
    ├── WithProvider(p)        → provider.Provider
    ├── WithModel(m)           → string (unified model setter)
    ├── WithTools(t...)        → tool.Registry
    ├── WithSkill(s)           → skill.Skill
    ├── WithMCP("npx ...")     → mcp.Client (Stdio or SSE, auto-detected)
    ├── WithMaxIter(n)         → int (default: 10)
    ├── WithThinkingLevel(l)   → provider.ThinkingLevel
    ├── WithProxy(cfg)         → provider.ProxyConfig
    └── Use(hook...)          → hook.Chain
```

### Core Components

| Package | Purpose |
|---------|---------|
| `agent/` | ReAct loop, Builder, message handling |
| `provider/` | Unified LLM interface + Anthropic/Gemini/OpenAI implementations |
| `tool/` | Tool interface, Registry, Schema builder, MCP wrapper, plan tool |
| `hook/` | Hook interface, Chain (fan-out), BaseHook, builtin Logger |

### Control Flow

1. `Agent.Run()` starts the ReAct loop
2. Each iteration: stream LLM response → parse tool calls → execute tools → append results
3. Hooks fire at: AgentStart, ThinkStart, ThinkEnd, ToolStart, ToolEnd, PlanCreated, AgentFinish, AgentError
4. Loop exits on `end_turn` stop reason or `maxIter` exceeded

## Thinking Level Feature

The thinking level feature allows you to control the depth of reasoning the LLM uses when processing requests:

```go
agent.New().
    WithProvider(anthropicprovider.New(apiKey, nil)).
    WithModel("claude-opus-4-7").
    WithThinkingLevel(provider.ThinkingLevelHigh). // thinking/reasoning depth level
    Build()
```

### Provider Implementations

| Provider | ThinkingLevelLow | ThinkingLevelMedium | ThinkingLevelHigh |
|----------|------------------|---------------------|-------------------|
| Anthropic | 2000 tokens | 5000 tokens | 10000 tokens |
| Gemini | IncludeThoughts: true | IncludeThoughts: true | IncludeThoughts: true |
| OpenAI | reasoning_effort: "low" | reasoning_effort: "medium" | reasoning_effort: "high" |

## Provider Interface

Every provider must implement:
```go
type Provider interface {
    Name() string
    Model() string
    SetModel(model string)
    Chat(ctx context.Context, req *Request) (*Response, error)
    Stream(ctx context.Context, req *Request) (<-chan *Chunk, error)
}
```

### Request Structure
```go
type Request struct {
    System        string
    Messages      []Message
    Tools         []ToolDef
    Stream        bool
    ThinkingLevel ThinkingLevel // thinking/reasoning depth level
}
```

### Thinking Levels
```go
type ThinkingLevel string

const (
    ThinkingLevelNone   ThinkingLevel = ""    // No thinking (default)
    ThinkingLevelLow    ThinkingLevel = "low"    // Low thinking depth
    ThinkingLevelMedium ThinkingLevel = "medium" // Medium thinking depth
    ThinkingLevelHigh   ThinkingLevel = "high"   // High thinking depth
)
```

**Default models**: Anthropic=`claude-opus-4-7`, Gemini=`gemini-2.0-flash`, OpenAI=`gpt-4o`

## Tool Interface

```go
type Tool interface {
    Name() string
    Description() string
    InputSchema() json.RawMessage
    Run(ctx context.Context, input json.RawMessage) (string, error)
}
```

Use `tool.NewSchema()` for fluent JSON Schema building:
```go
tool.NewSchema().String("query", "Description", true).Int("count", "...", false).Build()
```

**Registry panics on duplicate names** (no deduplication).

## Builder Pattern

The Builder pattern provides a fluent API for configuring agents:

```go
agent.New().
    WithProvider(anthropicprovider.New(apiKey, nil)).
    WithModel("claude-opus-4-7").
    WithTools(tool1, tool2).
    WithSkill(mySkill).
    WithMCP("npx -y @modelcontextprotocol/server-filesystem /tmp").
    WithMaxIter(10).
    WithThinkingLevel(provider.ThinkingLevelMedium).
    WithProxy(provider.ProxyConfig{HTTPProxy: "http://proxy:8080"}).
    Use(hook1, hook2).
    Build()
```

## Hook System

Embed `hook.BaseHook` and override only needed methods:

```go
type MyHook struct{ hook.BaseHook }
func (h *MyHook) OnToolStart(ctx context.Context, e *hook.ToolStartEvent) error {
    e.Input = mutate(e.Input) // hooks CAN modify input
    return nil
}
```

**ToolStartEvent.Input** and **ToolEndEvent.Output** are mutable — hooks transform tool I/O.

## MCP Integration

Auto-detects transport from spec:
- `npx -y @modelcontextprotocol/server-filesystem /tmp` → **Stdio** (local process)
- `https://my-mcp-server.com` → **SSE** (HTTP)

MCP tools discovered lazily at `Build()` time with background context.

## Known Issues

- No known issues at this time.

## Code Conventions

- Interfaces in `provider/provider.go`, `tool/tool.go`, `hook/hook.go`
- Implementations in subdirectories (e.g., `provider/anthropic/`, `tool/mcp/`)
- Builder pattern for `Agent` configuration
- `sync.RWMutex` for concurrent tool registry access
- SSE chunked responses use last chunk to carry assembled JSON
