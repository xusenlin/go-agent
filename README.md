# go-agent

[![CI](https://github.com/xusenlin/go-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/xusenlin/go-agent/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/xusenlin/go-agent.svg)](https://pkg.go.dev/github.com/xusenlin/go-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A clean, extensible ReAct agent in Go with streaming LLM support, a composable
hook system, Skill bundles, and MCP (Model Context Protocol) integration.

## Features

| Capability | Details |
|---|---|
| **Providers** | Anthropic, Google Gemini, OpenAI-compatible (DeepSeek, Moonshot…) |
| **Hook system** | Chainable middleware via `.Use()` — intercept & mutate tool I/O |
| **Streaming** | LLM responses streamed internally; hooks see the full assembled response |
| **Skills** | Bundle tools + system-prompt fragments into reusable packages |
| **MCP** | Auto-discover tools from any MCP server (Stdio or SSE transport) |
| **Proxy** | Single `WithProxy()` call covers HTTP clients *and* child-process env vars |

---

## Quick start

```bash
go get github.com/xusenlin/go-agent
```

```go
package main

import (
    "context"
    "fmt"
    "log/slog"
    "os"

    "github.com/xusenlin/go-agent/agent"
    "github.com/xusenlin/go-agent/hook/builtin"
    anthropicprovider "github.com/xusenlin/go-agent/provider/anthropic"
    "github.com/xusenlin/go-agent/tool/plan"
)

func main() {
    a, _ := agent.New().
        WithProvider(anthropicprovider.New(os.Getenv("ANTHROPIC_API_KEY"), nil)).
        WithModel("claude-opus-4-7").  // unified setter — works for any provider
        WithTools(plan.New()).
        WithMaxIter(10).
        Use(builtin.NewLogger(slog.Default())).
        Build()

    result, _ := a.Run(context.Background(), "Analyse the pros and cons of Go generics.")
    fmt.Println(result.Output)
}
```

### Default models per provider

| Provider | Default | Change with |
|---|---|---|
| Anthropic | `claude-opus-4-7` | `.WithModel("claude-sonnet-4-6")` |
| Gemini | `gemini-2.0-flash` | `.WithModel("gemini-2.5-pro")` |
| OpenAI | `gpt-4o` | `.WithModel("deepseek-chat")` |

---

## Architecture

```
agent.New()  (Builder)
    │
    ├── WithProvider(p)        → provider.Provider
    │                            ├── anthropic  (official SDK)
    │                            ├── gemini     (official SDK)
    │                            └── openai     (hand-rolled, zero deps)
    │
    ├── WithTools(t...)        → tool.Registry
    │                            ├── Native tools (implement tool.Tool)
    │                            └── MCP tools   (auto-wrapped)
    │
    ├── WithSkill(s)           → skill.Skill
    │                            └── Tools() + SystemPrompt() merged in
    │
    ├── WithMCP("npx ...")     → mcp.Client (Stdio or SSE, auto-detected)
    │
    ├── WithProxy(cfg)         → provider.ProxyConfig
    │                            ├── http.Client for all HTTP calls
    │                            └── os.Environ() inject for child procs
    │
    └── Use(hook...)           → hook.Chain
                                 ├── OnAgentStart
                                 ├── OnThinkStart / OnThinkEnd
                                 ├── OnToolStart  ← can mutate input
                                 ├── OnToolEnd    ← can mutate output
                                 ├── OnPlanCreated
                                 ├── OnAgentFinish
                                 └── OnAgentError
```

---

## Writing a custom hook

Embed `hook.BaseHook` and override only what you need:

```go
type MyHook struct{ hook.BaseHook }

// Intercept tool input before execution
func (h *MyHook) OnToolStart(ctx context.Context, e *hook.ToolStartEvent) error {
    fmt.Println("about to call:", e.ToolName)
    return nil
}

// Mutate tool output before it goes back to the LLM
func (h *MyHook) OnToolEnd(ctx context.Context, e *hook.ToolEndEvent) error {
    e.Output = "[sanitised] " + e.Output
    return nil
}

// React to a plan being created (e.g. render UI)
func (h *MyHook) OnPlanCreated(ctx context.Context, e *hook.PlanCreatedEvent) error {
    for i, step := range e.Steps {
        fmt.Printf("%d. %s\n", i+1, step.Title)
    }
    return nil
}
```

Register it with:

```go
agent.New().Use(&MyHook{})...
```

---

## Writing a custom tool

```go
type MyTool struct{}

func (t *MyTool) Name() string        { return "my_tool" }
func (t *MyTool) Description() string { return "Does something useful." }
func (t *MyTool) InputSchema() json.RawMessage {
    return tool.NewSchema().
        String("query", "What to look up", true).
        Build()
}
func (t *MyTool) Run(ctx context.Context, raw json.RawMessage) (string, error) {
    var in struct{ Query string `json:"query"` }
    json.Unmarshal(raw, &in)
    return "result for: " + in.Query, nil
}
```

---

## MCP integration

```go
// Stdio (local process — auto-detected from non-URL spec)
agent.New().WithMCP("npx -y @modelcontextprotocol/server-filesystem /tmp")

// SSE (remote server — auto-detected from http/https prefix)
agent.New().WithMCP("https://my-mcp-server.example.com")

// With proxy (env vars are injected into the child process automatically)
agent.New().
    WithProxy(provider.ProxyConfig{HTTPSProxy: "http://127.0.0.1:7890"}).
    WithMCP("npx -y @modelcontextprotocol/server-brave-search")
```

---

## Running the examples

```bash
# Plan demo (Anthropic)
ANTHROPIC_API_KEY=sk-... go run ./examples/plan_demo

# MCP filesystem demo (OpenAI-compatible / DeepSeek)
OPENAI_API_KEY=sk-... \
OPENAI_BASE_URL=https://api.deepseek.com/v1 \
go run ./examples/mcp_demo
```

---

## Dependency setup

```bash
go mod tidy
# or individually:
go get github.com/anthropics/anthropic-sdk-go
go get google.golang.org/genai
```

---

## License

MIT © xusenlin. See [LICENSE](./LICENSE).
