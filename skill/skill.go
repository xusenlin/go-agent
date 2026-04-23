package skill

import "github.com/xusenlin/go-agent/tool"

// Skill is a composable unit that can add tools and system prompt fragments
// to the agent. Embed it in a concrete struct and implement the methods.
type Skill interface {
	// Name is the unique identifier for this skill.
	Name() string
	// SystemPrompt returns an optional fragment appended to the system prompt
	// when this skill is registered.
	SystemPrompt() string
	// Tools returns any tools this skill provides. May be nil/empty.
	Tools() []tool.Tool
}

// BaseSkill provides no-op implementations for all Skill methods.
// Embed it in your struct and override only what you need.
type BaseSkill struct{}

func (BaseSkill) Name() string         { return "" }
func (BaseSkill) SystemPrompt() string { return "" }
func (BaseSkill) Tools() []tool.Tool   { return nil }
