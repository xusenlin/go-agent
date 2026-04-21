package builtin

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/xusenlin/go-agent/hook"
)

// Logger is a built-in hook that logs every agent lifecycle event via slog.
// It is read-only and never modifies any event fields.
type Logger struct {
	hook.BaseHook
	log *slog.Logger
}

// NewLogger creates a Logger hook. Pass slog.Default() for the default logger.
func NewLogger(log *slog.Logger) *Logger {
	if log == nil {
		log = slog.Default()
	}
	return &Logger{log: log}
}

func (l *Logger) OnAgentStart(_ context.Context, e *hook.AgentStartEvent) error {
	l.log.Info("[agent] start", "input", truncate(e.Input, 120))
	return nil
}

func (l *Logger) OnThinkStart(_ context.Context, e *hook.ThinkStartEvent) error {
	l.log.Debug("[agent] thinking", "iteration", e.Iteration, "messages", len(e.Messages))
	return nil
}

func (l *Logger) OnThinkEnd(_ context.Context, e *hook.ThinkEndEvent) error {
	l.log.Info("[agent] think end",
		"iteration", e.Iteration,
		"stop_reason", e.Response.StopReason,
		"tool_calls", len(e.Response.ToolCalls),
		"content", truncate(e.Response.Content, 120),
	)
	return nil
}

func (l *Logger) OnToolStart(_ context.Context, e *hook.ToolStartEvent) error {
	l.log.Info("[tool] start",
		"iteration", e.Iteration,
		"tool", e.ToolName,
		"call_id", e.CallID,
		"input", truncateJSON(e.Input, 200),
	)
	return nil
}

func (l *Logger) OnToolEnd(_ context.Context, e *hook.ToolEndEvent) error {
	if e.Err != nil {
		l.log.Error("[tool] error",
			"iteration", e.Iteration,
			"tool", e.ToolName,
			"call_id", e.CallID,
			"err", e.Err,
		)
	} else {
		l.log.Info("[tool] end",
			"iteration", e.Iteration,
			"tool", e.ToolName,
			"call_id", e.CallID,
			"output", truncate(e.Output, 200),
		)
	}
	return nil
}

func (l *Logger) OnPlanCreated(_ context.Context, e *hook.PlanCreatedEvent) error {
	titles := make([]string, len(e.Steps))
	for i, s := range e.Steps {
		titles[i] = s.Title
	}
	l.log.Info("[plan] created",
		"iteration", e.Iteration,
		"steps", titles,
	)
	return nil
}

func (l *Logger) OnAgentFinish(_ context.Context, e *hook.AgentFinishEvent) error {
	l.log.Info("[agent] finish",
		"iterations", e.Iterations,
		"output", truncate(e.Output, 200),
	)
	return nil
}

func (l *Logger) OnAgentError(_ context.Context, e *hook.AgentErrorEvent) error {
	l.log.Error("[agent] error",
		"iterations", e.Iterations,
		"err", e.Err,
	)
	return nil
}

// ─── helpers ──────────────────────────────────────────────────────────────────

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func truncateJSON(raw json.RawMessage, n int) string {
	return truncate(string(raw), n)
}
