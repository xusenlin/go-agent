// Package agent implements a ReAct (Reason + Act) loop with a composable
// hook system, multi-provider support, skills, and MCP tool integration.
package agent

import (
	"errors"
	"fmt"
	"strings"

	"github.com/xusenlin/go-agent/hook"
	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/skill"
	"github.com/xusenlin/go-agent/tool"
	"github.com/xusenlin/go-agent/tool/mcp"
)

const defaultMaxIter = 10
const defaultSystemPrompt = `You are a helpful AI assistant.
When given a task, think step by step.
Use the available tools to accomplish the task.
When you have enough information to answer, respond directly without calling any more tools.`

// Agent is the assembled, ready-to-run agent.
// Create one via Builder.Build() and call Run() to start the ReAct loop.
type Agent struct {
	provider      provider.Provider
	registry      *tool.Registry
	hooks         *hook.Chain
	systemPrompt  string
	maxIter       int
	thinkingLevel provider.ThinkingLevel
	proxy         *provider.ProxyConfig
	mcpClients    []*mcp.Client
	history       []provider.Message // conversation history before the current input
}

// Close releases resources held by the agent, including MCP client connections.
func (a *Agent) Close() error {
	var errs []error
	for _, c := range a.mcpClients {
		if err := c.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("agent: close MCP clients: %v", errs)
	}
	return nil
}

// ─── Builder ──────────────────────────────────────────────────────────────────

// Builder constructs an Agent through a fluent API.
// Start with New() and finish with Build().
type Builder struct {
	provider      provider.Provider
	model         string // optional override applied to provider at Build() time
	tools         []tool.Tool
	skills        []skill.Skill
	mcpSpecs      []string
	hooks         []hook.Hook
	systemPrompt  string
	maxIter       int
	thinkingLevel provider.ThinkingLevel // thinking/reasoning depth level
	proxy         *provider.ProxyConfig
	history       []provider.Message // conversation history before the current input
	errs          []error
}

// New returns a fresh Builder.
func New() *Builder {
	return &Builder{
		maxIter:      defaultMaxIter,
		systemPrompt: defaultSystemPrompt,
	}
}

// WithProvider sets the LLM backend. Required.
func (b *Builder) WithProvider(p provider.Provider) *Builder {
	b.provider = p
	return b
}

// WithModel overrides the model on whatever provider was passed to WithProvider.
// This is the recommended way to specify a model — you don't need to remember
// each provider's specific WithModel option.
//
// Example:
//
//	agent.New().
//	    WithProvider(anthropic.New(key, nil)).
//	    WithModel("claude-sonnet-4-6").   // ← unified across all providers
//	    Build()
func (b *Builder) WithModel(model string) *Builder {
	b.model = model
	return b
}

// WithTools registers one or more native tools.
func (b *Builder) WithTools(tools ...tool.Tool) *Builder {
	b.tools = append(b.tools, tools...)
	return b
}

// WithSkill registers a skill, adding its tools and appending its system
// prompt fragment.
func (b *Builder) WithSkill(s skill.Skill) *Builder {
	b.skills = append(b.skills, s)
	return b
}

// WithMCP registers an MCP server by spec (command or URL).
// Tools are discovered lazily at Build() time.
func (b *Builder) WithMCP(spec string) *Builder {
	b.mcpSpecs = append(b.mcpSpecs, spec)
	return b
}

// WithSystemPrompt overrides the default system prompt.
func (b *Builder) WithSystemPrompt(prompt string) *Builder {
	b.systemPrompt = prompt
	return b
}

// WithMaxIter sets the maximum number of ReAct iterations before aborting.
func (b *Builder) WithMaxIter(n int) *Builder {
	b.maxIter = n
	return b
}

// WithThinkingLevel sets the thinking/reasoning depth level for the LLM.
// Options: provider.ThinkingLevelNone (default), provider.ThinkingLevelLow,
// provider.ThinkingLevelMedium, provider.ThinkingLevelHigh.
func (b *Builder) WithThinkingLevel(level provider.ThinkingLevel) *Builder {
	b.thinkingLevel = level
	return b
}

// WithProxy sets an HTTP proxy for all outbound connections (LLM API calls,
// MCP SSE connections, and the env vars injected into MCP stdio processes).
func (b *Builder) WithProxy(proxy provider.ProxyConfig) *Builder {
	b.proxy = &proxy
	return b
}

// WithHistory sets the conversation history to include before the current input.
// This allows the agent to have context from previous messages in the conversation.
func (b *Builder) WithHistory(history []provider.Message) *Builder {
	b.history = history
	return b
}

// Use adds one or more hooks to the chain. Hooks fire in registration order.
func (b *Builder) Use(hooks ...hook.Hook) *Builder {
	b.hooks = append(b.hooks, hooks...)
	return b
}

// Build validates the configuration, discovers MCP tools, and returns a
// ready-to-use Agent. Returns an error if the provider is missing or if
// any MCP server is unreachable.
func (b *Builder) Build() (*Agent, error) {
	if b.provider == nil {
		return nil, errors.New("agent: WithProvider is required")
	}

	// Validate maxIter
	if b.maxIter <= 0 {
		return nil, errors.New("agent: maxIter must be positive")
	}

	// Apply model override, if any
	if b.model != "" {
		b.provider.SetModel(b.model)
	}

	// Assemble system prompt from base + skill fragments
	systemParts := []string{b.systemPrompt}
	for _, s := range b.skills {
		if sp := s.SystemPrompt(); sp != "" {
			systemParts = append(systemParts, "\n## Skill: "+s.Name()+"\n"+sp)
		}
	}

	// Build tool registry
	reg := tool.NewRegistry()

	// Native tools
	if len(b.tools) > 0 {
		reg.Register(b.tools...)
	}

	// Skill tools
	for _, s := range b.skills {
		if len(s.Tools()) > 0 {
			reg.Register(s.Tools()...)
		}
	}

	// MCP tools (discovered at build time, using background context)
	// The context is short-lived; actual tool calls use the Run() context.
	var mcpClients []*mcp.Client
	for _, spec := range b.mcpSpecs {
		mcpTools, client, err := mcp.DiscoverTools(b.buildCtx(), spec, b.proxy)
		if err != nil {
			// Close any previously opened MCP clients
			for _, c := range mcpClients {
				_ = c.Close()
			}
			return nil, errors.Join(errors.New("agent: MCP discover failed for "+spec), err)
		}
		reg.Register(mcpTools...)
		mcpClients = append(mcpClients, client)
	}

	// Build hook chain
	chain := hook.NewChain()
	chain.Add(b.hooks...)

	return &Agent{
		provider:      b.provider,
		registry:      reg,
		hooks:         chain,
		systemPrompt:  strings.Join(systemParts, "\n"),
		maxIter:       b.maxIter,
		thinkingLevel: b.thinkingLevel,
		proxy:         b.proxy,
		mcpClients:    mcpClients,
		history:       b.history,
	}, nil
}
