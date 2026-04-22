// Package hook defines the Hook interface and all event types used by the agent.
//
// Usage pattern:
//
//	type MyHook struct{ hook.BaseHook }
//
//	func (h *MyHook) OnToolStart(ctx context.Context, e *hook.ToolStartEvent) error {
//	    fmt.Println("calling tool:", e.ToolName)
//	    return nil
//	}
package hook

import (
	"context"
	"encoding/json"

	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/tool/plan"
)

// ─── Event types ──────────────────────────────────────────────────────────────

// AgentStartEvent fires once when Agent.Run is called.
type AgentStartEvent struct {
	Input string // the user's original query
}

// ThinkStartEvent fires just before the LLM is called.
type ThinkStartEvent struct {
	Iteration int
	Messages  []provider.Message // snapshot of current history
}

// ThinkEndEvent fires after the LLM returns a full response.
// Hooks may not modify the response; it is read-only observation.
type ThinkEndEvent struct {
	Iteration int
	Response  *provider.Response
	// StreamTokens is the full text assembled from streaming chunks.
	StreamTokens string
}

// ToolStartEvent fires before a tool is executed.
// Hooks MAY modify Input to intercept/transform the tool's arguments.
type ToolStartEvent struct {
	Iteration int
	ToolName  string
	CallID    string
	Input     json.RawMessage // MUTABLE: hooks can replace this
}

// ToolEndEvent fires after a tool returns.
// Hooks MAY modify Output to transform the tool's result before it is sent to the LLM.
type ToolEndEvent struct {
	Iteration int
	ToolName  string
	CallID    string
	Output    string // MUTABLE: hooks can replace this
	Err       error  // if non-nil, the tool failed
}

// PlanCreatedEvent fires when the plan tool produces a plan.
// This is a specialisation of ToolEndEvent for the create_plan tool,
// making it easy for UI hooks to render a progress view.
type PlanCreatedEvent struct {
	Iteration int
	Steps     []plan.Step
}

// AgentFinishEvent fires when the agent produces a final answer.
type AgentFinishEvent struct {
	Output     string
	Iterations int
}

// AgentErrorEvent fires when the agent aborts due to a fatal error.
type AgentErrorEvent struct {
	Err        error
	Iterations int
}

// ─── Hook interface ───────────────────────────────────────────────────────────

// Hook is implemented by anything that wants to observe or intercept the
// agent's ReAct loop. Embed BaseHook and override only the methods you care about.
type Hook interface {
	// OnAgentStart fires once before the ReAct loop begins.
	OnAgentStart(ctx context.Context, e *AgentStartEvent) error

	// OnThinkStart fires just before each LLM call.
	OnThinkStart(ctx context.Context, e *ThinkStartEvent) error

	// OnThinkEnd fires after each LLM call returns.
	OnThinkEnd(ctx context.Context, e *ThinkEndEvent) error

	// OnToolStart fires before a tool is executed.
	// The hook MAY modify e.Input to change the tool's arguments.
	OnToolStart(ctx context.Context, e *ToolStartEvent) error

	// OnToolEnd fires after a tool returns.
	// The hook MAY modify e.Output to change what gets sent back to the LLM.
	OnToolEnd(ctx context.Context, e *ToolEndEvent) error

	// OnPlanCreated fires specifically when the create_plan tool succeeds.
	// It fires in addition to (and after) OnToolEnd.
	OnPlanCreated(ctx context.Context, e *PlanCreatedEvent) error

	// OnAgentFinish fires when the agent produces its final answer.
	OnAgentFinish(ctx context.Context, e *AgentFinishEvent) error

	// OnAgentError fires if the agent aborts with an error.
	OnAgentError(ctx context.Context, e *AgentErrorEvent) error
}

// ─── BaseHook ─────────────────────────────────────────────────────────────────

// BaseHook provides no-op implementations of every Hook method.
// Embed it in your struct and override only what you need.
type BaseHook struct{}

func (BaseHook) OnAgentStart(_ context.Context, _ *AgentStartEvent) error  { return nil }
func (BaseHook) OnThinkStart(_ context.Context, _ *ThinkStartEvent) error  { return nil }
func (BaseHook) OnThinkEnd(_ context.Context, _ *ThinkEndEvent) error      { return nil }
func (BaseHook) OnToolStart(_ context.Context, _ *ToolStartEvent) error    { return nil }
func (BaseHook) OnToolEnd(_ context.Context, _ *ToolEndEvent) error        { return nil }
func (BaseHook) OnPlanCreated(_ context.Context, _ *PlanCreatedEvent) error { return nil }
func (BaseHook) OnAgentFinish(_ context.Context, _ *AgentFinishEvent) error { return nil }
func (BaseHook) OnAgentError(_ context.Context, _ *AgentErrorEvent) error  { return nil }
