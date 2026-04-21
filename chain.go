package hook

import (
	"context"
	"errors"
)

// Chain holds an ordered list of hooks and fans out every event to all of them.
// Errors from individual hooks are collected with errors.Join; execution
// continues for all remaining hooks even if one fails.
type Chain struct {
	hooks []Hook
}

// NewChain creates an empty Chain.
func NewChain() *Chain { return &Chain{} }

// Add appends one or more hooks to the chain.
func (c *Chain) Add(hooks ...Hook) *Chain {
	c.hooks = append(c.hooks, hooks...)
	return c
}

// Len returns the number of hooks in the chain.
func (c *Chain) Len() int { return len(c.hooks) }

// ─── Dispatch methods ─────────────────────────────────────────────────────────
// Each method calls the corresponding Hook method on every registered hook.
// Errors are accumulated and returned via errors.Join.

func (c *Chain) OnAgentStart(ctx context.Context, e *AgentStartEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnAgentStart(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnThinkStart(ctx context.Context, e *ThinkStartEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnThinkStart(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnThinkEnd(ctx context.Context, e *ThinkEndEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnThinkEnd(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnToolStart(ctx context.Context, e *ToolStartEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnToolStart(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnToolEnd(ctx context.Context, e *ToolEndEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnToolEnd(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnPlanCreated(ctx context.Context, e *PlanCreatedEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnPlanCreated(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnAgentFinish(ctx context.Context, e *AgentFinishEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnAgentFinish(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Chain) OnAgentError(ctx context.Context, e *AgentErrorEvent) error {
	var errs []error
	for _, h := range c.hooks {
		if err := h.OnAgentError(ctx, e); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
