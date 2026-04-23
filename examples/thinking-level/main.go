// examples/thinking-level/main.go
//
// Demonstrates the thinking level feature:
//   - OpenAI provider with thinking level
//   - Shows how thinking level affects the agent's reasoning
//
// Usage:
//
//	go run ./examples/thinking-level
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/xusenlin/go-agent/agent"
	"github.com/xusenlin/go-agent/config"
	"github.com/xusenlin/go-agent/provider"
	openai "github.com/xusenlin/go-agent/provider/openai"
)

func main() {
	// Load environment variables from .env file
	config.LoadEnv()

	apiKey, model, baseURL := config.GetOpenAIConfig()
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set")
	}

	ctx := context.Background()

	// Build agent with thinking level
	a, err := agent.New().
		WithProvider(openai.New(apiKey, nil, openai.WithBaseURL(baseURL))).
		WithModel(model).
		WithThinkingLevel(provider.ThinkingLevelHigh). // thinking/reasoning depth level
		Build()
	if err != nil {
		log.Fatalf("build error: %v", err)
	}

	// Run with a complex reasoning task
	query := "Explain the concept of recursion in programming, provide examples, and discuss its advantages and disadvantages."
	fmt.Println("Query:", query)
	fmt.Println("Thinking budget: 10000 tokens")
	fmt.Println("────────────────────────────────────────────────────────────")

	result, err := a.Run(ctx, query)
	if err != nil {
		log.Fatalf("run error: %v", err)
	}

	fmt.Println("────────────────────────────────────────────────────────────")
	fmt.Println("Final answer:")
	fmt.Println(result.Output)
	fmt.Printf("\n(completed in %d iteration(s))\n", result.Iterations)
}
