package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
)

// ─── Shared message types ────────────────────────────────────────────────────

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is a single turn in the conversation.
type Message struct {
	Role        Role
	Content     string
	ToolCalls   []ToolCall   // assistant → tool calls
	ToolResults []ToolResult // user → tool results
}

// ToolCall is a single tool invocation requested by the LLM.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult is the response we send back for a ToolCall.
type ToolResult struct {
	ToolCallID string
	Content    string
	IsError    bool
}

// ToolDef describes a tool to the LLM (JSON Schema).
type ToolDef struct {
	Name        string
	Description string
	InputSchema json.RawMessage // standard JSON Schema object
}

// ─── Request / Response ──────────────────────────────────────────────────────

// ThinkingLevel represents the depth of reasoning/thinking for the LLM.
type ThinkingLevel string

const (
	ThinkingLevelNone   ThinkingLevel = ""       // No thinking (default)
	ThinkingLevelLow    ThinkingLevel = "low"    // Low thinking depth
	ThinkingLevelMedium ThinkingLevel = "medium" // Medium thinking depth
	ThinkingLevelHigh   ThinkingLevel = "high"   // High thinking depth
)

// Request is the unified chat request across all providers.
type Request struct {
	System        string
	Messages      []Message
	Tools         []ToolDef
	Stream        bool          // hint; streaming is always handled via Stream()
	ThinkingLevel ThinkingLevel // thinking/reasoning depth level (empty = disabled)
}

// Response is the fully assembled LLM response (non-streaming or assembled).
type Response struct {
	Content    string
	Thinking   string    // thinking/reasoning content
	ToolCalls  []ToolCall
	StopReason string // "end_turn" | "tool_use" | "max_tokens"
	Usage      Usage  // token usage stats (may be zero if unavailable)
}

// Usage holds token usage statistics for a request.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

// Chunk is a single streaming delta from the LLM.
type Chunk struct {
	Delta        string // text delta
	ToolCallID   string // non-empty when this chunk carries tool call info
	ToolName     string
	InputDelta   string // partial JSON for tool input
	Done         bool
	InputTokens  int    // cumulative input tokens for this request
	OutputTokens int    // cumulative output tokens for this request
	TotalTokens  int    // total tokens (input + output)
	IsThinking   bool   // true if this chunk contains thinking/reasoning content
	Error        string // non-empty when an error occurred
}

// ─── Provider interface ───────────────────────────────────────────────────────

// Provider is the unified interface every LLM backend must implement.
type Provider interface {
	// Name returns the provider identifier, e.g. "anthropic", "google".
	Name() string

	// Model returns the currently configured model name.
	Model() string

	// SetModel overrides the model name at runtime.
	// Called by agent.Builder.WithModel() for unified configuration.
	SetModel(model string)

	// Chat sends a request and waits for the full response.
	Chat(ctx context.Context, req *Request) (*Response, error)

	// Stream sends a request and returns a channel of incremental chunks.
	// The caller must drain the channel until Chunk.Done == true or ctx is cancelled.
	// The final assembled Response is also available via the last Chunk (Done==true)
	// but callers that only need the final result should prefer Chat().
	Stream(ctx context.Context, req *Request) (<-chan *Chunk, error)
}

// ─── Proxy config ─────────────────────────────────────────────────────────────

// ProxyConfig holds HTTP/HTTPS proxy settings applied to all outbound requests.
type ProxyConfig struct {
	HTTPProxy  string   // e.g. "http://127.0.0.1:7890"
	HTTPSProxy string   // e.g. "http://127.0.0.1:7890"
	NoProxy    []string // hostnames to bypass
}

// HTTPClient builds an *http.Client that routes traffic through the proxy.
func (p *ProxyConfig) HTTPClient() *http.Client {
	if p == nil {
		return http.DefaultClient
	}
	proxyFunc := func(req *http.Request) (*url.URL, error) {
		host := req.URL.Hostname()
		for _, np := range p.NoProxy {
			if host == np {
				return nil, nil
			}
		}
		raw := p.HTTPProxy
		if req.URL.Scheme == "https" && p.HTTPSProxy != "" {
			raw = p.HTTPSProxy
		}
		if raw == "" {
			return nil, nil
		}
		return url.Parse(raw)
	}
	return &http.Client{
		Transport: &http.Transport{Proxy: proxyFunc},
	}
}

// Environ returns a slice of "KEY=VALUE" strings suitable for exec.Cmd.Env,
// so that child processes (e.g. MCP stdio servers) inherit the proxy settings.
func (p *ProxyConfig) Environ() []string {
	if p == nil {
		return nil
	}
	var env []string
	if p.HTTPProxy != "" {
		env = append(env, "HTTP_PROXY="+p.HTTPProxy, "http_proxy="+p.HTTPProxy)
	}
	if p.HTTPSProxy != "" {
		env = append(env, "HTTPS_PROXY="+p.HTTPSProxy, "https_proxy="+p.HTTPSProxy)
	}
	return env
}
