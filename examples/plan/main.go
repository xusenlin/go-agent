// examples/plan/main.go
//
// Demonstrates the ReAct agent with:
//   - OpenAI provider (streaming)
//   - Built-in plan tool
//   - A custom PlanUIHook that renders the plan to the terminal
//   - A simple calculator tool to show non-plan tool calls
//
// Usage:
//
//	go run ./examples/plan
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/xusenlin/go-agent/agent"
	"github.com/xusenlin/go-agent/config"
	"github.com/xusenlin/go-agent/hook"
	hookbuiltin "github.com/xusenlin/go-agent/hook/builtin"
	"github.com/xusenlin/go-agent/provider"
	openai "github.com/xusenlin/go-agent/provider/openai"
	"github.com/xusenlin/go-agent/tool"
	"github.com/xusenlin/go-agent/tool/plan"
)

func main() {
	// Load environment variables from .env file
	config.LoadEnv()

	apiKey, model, baseURL := config.GetOpenAIConfig()
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is not set")
		os.Exit(1)
	}

	ctx := context.Background()

	// ── Build agent ───────────────────────────────────────────────────────────
	a, err := agent.New().
		WithProvider(openai.New(apiKey, nil, openai.WithBaseURL(baseURL))).
		WithModel(model).
		WithTools(plan.New(), newCalculator()).
		WithMaxIter(10).
		WithThinkingLevel(provider.ThinkingLevelHigh).
		Use(hookbuiltin.NewLogger(slog.Default())).
		Use(&PlanUIHook{}).
		Build()
	if err != nil {
		fmt.Fprintln(os.Stderr, "build error:", err)
		os.Exit(1)
	}

	// ── Run ───────────────────────────────────────────────────────────────────
	query := "Calculate the sum of 42 and 58, then the product of those two numbers, and give me a final report."
	fmt.Println("Query:", query)
	fmt.Println(strings.Repeat("─", 60))

	result, err := a.Run(ctx, query)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run error:", err)
		os.Exit(1)
	}

	fmt.Println(strings.Repeat("─", 60))
	fmt.Println("Final answer:")
	fmt.Println(result.Output)
	fmt.Printf("\n(completed in %d iteration(s))\n", result.Iterations)
}

// ─── PlanUIHook ───────────────────────────────────────────────────────────────

// PlanUIHook renders the plan steps to the terminal when create_plan is called.
// Replace the fmt.Println calls with your actual UI framework (Bubble Tea, etc).
type PlanUIHook struct{ hook.BaseHook }

func (h *PlanUIHook) OnToolEnd(_ context.Context, e *hook.ToolEndEvent) error {
	if e.ToolName == "create_plan" {
		// Parse plan output
		var result struct {
			Title string `json:"title"`
			Steps []struct {
				Title       string `json:"title"`
				Description string `json:"description"`
				Status      string `json:"status"`
			} `json:"steps"`
		}
		if err := json.Unmarshal([]byte(e.Output), &result); err == nil {
			fmt.Println("\n📋 Plan created:", result.Title)
			for i, step := range result.Steps {
				fmt.Printf("  %d. [%s] %s\n", i+1, step.Status, step.Title)
				if step.Description != "" {
					fmt.Printf("     └─ %s\n", step.Description)
				}
			}
			fmt.Println()
		}
	} else {
		fmt.Printf("✅ Tool %q finished\n", e.ToolName)
	}
	return nil
}

// ─── Calculator tool ──────────────────────────────────────────────────────────

type calculator struct{}

func newCalculator() tool.Tool { return &calculator{} }

func (c *calculator) Name() string { return "calculator" }
func (c *calculator) Description() string {
	return "Perform basic arithmetic. Supported operations: add, subtract, multiply, divide."
}
func (c *calculator) InputSchema() json.RawMessage {
	return tool.NewSchema().
		String("operation", "One of: add, subtract, multiply, divide", true).
		Int("a", "First operand", true).
		Int("b", "Second operand", true).
		Build()
}

type calcInput struct {
	Operation string  `json:"operation"`
	A         float64 `json:"a"`
	B         float64 `json:"b"`
}

func (c *calculator) Run(_ context.Context, raw json.RawMessage) (string, error) {
	var in calcInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "", err
	}
	var result float64
	switch in.Operation {
	case "add":
		result = in.A + in.B
	case "subtract":
		result = in.A - in.B
	case "multiply":
		result = in.A * in.B
	case "divide":
		if in.B == 0 {
			return "", fmt.Errorf("division by zero")
		}
		result = in.A / in.B
	default:
		return "", fmt.Errorf("unknown operation: %q", in.Operation)
	}
	return fmt.Sprintf("%g", result), nil
}
