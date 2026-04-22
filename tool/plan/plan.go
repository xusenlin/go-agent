// Package plan provides a lightweight planning tool.
// When the LLM calls it, it emits a PlanCreatedEvent through the hook system
// so callers can render a UI, log, or otherwise react to the plan.
package plan

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/xusenlin/go-agent/tool"
)

// ToolName is the identifier the LLM uses to invoke the plan tool.
// Use this constant instead of hard-coding the string elsewhere.
const ToolName = "create_plan"

// Status represents the lifecycle of a single step.
type Status string

const (
	StatusPending    Status = "pending"
	StatusInProgress Status = "in_progress"
	StatusDone       Status = "done"
	StatusFailed     Status = "failed"
)

// Step is one item in a plan.
type Step struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Status      Status    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
}

// Plan is the tool the LLM calls to emit a structured plan.
type Plan struct{}

// New returns a ready-to-use Plan tool.
func New() *Plan { return &Plan{} }

func (p *Plan) Name() string { return ToolName }

func (p *Plan) Description() string {
	return "Create a structured step-by-step plan before starting work. " +
		"Call this tool early so the user can see what you intend to do."
}

func (p *Plan) InputSchema() json.RawMessage {
	return tool.NewSchema().
		String("title", "Short title for the overall plan", true).
		Array("steps", "Ordered list of step objects with 'title' and 'description' fields", "object", true).
		Build()
}

// input mirrors the JSON the LLM sends.
type input struct {
	Title string `json:"title"`
	Steps []struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	} `json:"steps"`
}

// Run parses the LLM's input and stores the steps in the context.
// The actual hook emission happens in agent/react.go after Run returns.
func (p *Plan) Run(_ context.Context, raw json.RawMessage) (string, error) {
	var in input
	if err := json.Unmarshal(raw, &in); err != nil {
		return "", err
	}
	// Return the plan as JSON so react.go can decode it and emit the hook event.
	steps := make([]Step, len(in.Steps))
	for i, s := range in.Steps {
		steps[i] = Step{
			ID:          fmt.Sprintf("step-%d", i+1),
			Title:       s.Title,
			Description: s.Description,
			Status:      StatusPending,
			CreatedAt:   time.Now(),
		}
	}
	result := map[string]any{
		"__plan__": true, // sentinel so react.go recognises this result
		"title":    in.Title,
		"steps":    steps,
	}
	out, _ := json.Marshal(result)
	return string(out), nil
}

// ParseResult tries to extract a Plan from the string returned by Run.
// Returns nil, false if the string is not a plan result.
func ParseResult(s string) ([]Step, bool) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, false
	}
	if _, ok := m["__plan__"]; !ok {
		return nil, false
	}
	var steps []Step
	if err := json.Unmarshal(m["steps"], &steps); err != nil {
		return nil, false
	}
	return steps, true
}

