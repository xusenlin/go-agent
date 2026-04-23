package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/xusenlin/go-agent/provider"
)

// Tool is the core interface every tool must implement.
type Tool interface {
	// Name is the unique identifier used by the LLM to call this tool.
	Name() string
	// Description explains to the LLM when and how to use the tool.
	Description() string
	// InputSchema returns a JSON Schema object describing the tool's parameters.
	InputSchema() json.RawMessage
	// Run executes the tool with the given JSON input and returns a string result.
	Run(ctx context.Context, input json.RawMessage) (string, error)
}

// ToProviderDef converts a Tool to the provider.ToolDef used in LLM requests.
func ToProviderDef(t Tool) provider.ToolDef {
	return provider.ToolDef{
		Name:        t.Name(),
		Description: t.Description(),
		InputSchema: t.InputSchema(),
	}
}

// ToProviderDefs converts a slice of Tools to provider.ToolDef slice.
func ToProviderDefs(tools []Tool) []provider.ToolDef {
	defs := make([]provider.ToolDef, len(tools))
	for i, t := range tools {
		defs[i] = ToProviderDef(t)
	}
	return defs
}

// ─── Registry ─────────────────────────────────────────────────────────────────

// Registry holds a named set of tools and provides safe concurrent lookup.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds one or more tools to the registry.
// Panics if a tool with the same name is already registered.
func (r *Registry) Register(tools ...Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, t := range tools {
		if _, exists := r.tools[t.Name()]; exists {
			panic(fmt.Sprintf("tool registry: duplicate tool name %q", t.Name()))
		}
		r.tools[t.Name()] = t
	}
}

// Get looks up a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// All returns all registered tools.
func (r *Registry) All() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// ─── Schema helpers ───────────────────────────────────────────────────────────

// SchemaBuilder is a fluent JSON Schema builder for tool input schemas.
type SchemaBuilder struct {
	schema map[string]any
	props  map[string]any
	req    []string
}

func NewSchema() *SchemaBuilder {
	return &SchemaBuilder{
		schema: map[string]any{"type": "object"},
		props:  map[string]any{},
	}
}

func (s *SchemaBuilder) String(name, desc string, required bool) *SchemaBuilder {
	s.props[name] = map[string]any{"type": "string", "description": desc}
	if required {
		s.req = append(s.req, name)
	}
	return s
}

func (s *SchemaBuilder) Int(name, desc string, required bool) *SchemaBuilder {
	s.props[name] = map[string]any{"type": "integer", "description": desc}
	if required {
		s.req = append(s.req, name)
	}
	return s
}

func (s *SchemaBuilder) Array(name, desc, itemType string, required bool) *SchemaBuilder {
	s.props[name] = map[string]any{
		"type":        "array",
		"description": desc,
		"items":       map[string]any{"type": itemType},
	}
	if required {
		s.req = append(s.req, name)
	}
	return s
}

func (s *SchemaBuilder) Build() json.RawMessage {
	s.schema["properties"] = s.props
	if len(s.req) > 0 {
		s.schema["required"] = s.req
	}
	raw, err := json.Marshal(s.schema)
	if err != nil {
		// This should never happen with a valid schema, but handle it gracefully
		return json.RawMessage(`{"type": "object"}`)
	}
	return raw
}
