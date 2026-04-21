// Package skill defines the Skill interface.
//
// A Skill is a reusable capability bundle that packages together:
//   - A set of tools the LLM can call
//   - An optional system-prompt fragment that is appended to the agent's system prompt
//
// Skills let you ship cohesive feature sets (e.g. "web research", "code review")
// that users can drop in with a single .WithSkill() call.
package skill

import "github.com/xusenlin/go-agent/tool"

// Skill is a composable capability bundle.
type Skill interface {
	// Name is a human-readable identifier, e.g. "web_research".
	Name() string

	// SystemPrompt returns an optional fragment appended to the agent's
	// system prompt to give the LLM context about when/how to use this skill.
	// Return "" if no extra instructions are needed.
	SystemPrompt() string

	// Tools returns the tools this skill exposes to the LLM.
	Tools() []tool.Tool
}

// ─── BaseSkill ────────────────────────────────────────────────────────────────

// BaseSkill provides sensible defaults. Embed and override as needed.
type BaseSkill struct {
	name         string
	systemPrompt string
	tools        []tool.Tool
}

// NewBase creates a BaseSkill with the given name, optional system prompt, and tools.
func NewBase(name, systemPrompt string, tools ...tool.Tool) *BaseSkill {
	return &BaseSkill{name: name, systemPrompt: systemPrompt, tools: tools}
}

func (s *BaseSkill) Name() string         { return s.name }
func (s *BaseSkill) SystemPrompt() string { return s.systemPrompt }
func (s *BaseSkill) Tools() []tool.Tool   { return s.tools }
