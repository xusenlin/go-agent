package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/xusenlin/go-agent/agent"
	"github.com/xusenlin/go-agent/config"
	openai "github.com/xusenlin/go-agent/provider/openai"
	"github.com/xusenlin/go-agent/tool"
)

func main() {
	// Load environment variables from .env file
	config.LoadEnv()

	apiKey, model, baseURL := config.GetOpenAIConfig()
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set")
	}

	ctx := context.Background()

	a, err := agent.New().
		WithProvider(openai.New(apiKey, nil, openai.WithBaseURL(baseURL))).
		WithModel(model).
		WithTools(newWeatherTool()).
		WithMaxIter(5).
		Build()
	if err != nil {
		log.Fatalf("build error: %v", err)
	}

	// ── Run with streaming ─────────────────────────────────────────────────────
	query := "What's the weather in Tokyo and Beijing? Compare them."
	fmt.Println("Query:", query)
	fmt.Println(strings.Repeat("─", 60))

	blocks, err := a.RunStream(ctx, query)
	if err != nil {
		log.Fatalf("run error: %v", err)
	}

	// ── Process streamed blocks ─────────────────────────────────────────────────
	for block := range blocks {
		printBlock(block)
	}

	fmt.Println(strings.Repeat("─", 60))
	fmt.Println("Stream completed.")
}

// ─── Weather tool ──────────────────────────────────────────────────────────────

type weatherTool struct{}

func newWeatherTool() tool.Tool { return &weatherTool{} }

func (t *weatherTool) Name() string { return "get_weather" }

func (t *weatherTool) Description() string {
	return "Get the current weather for a specified city. Returns temperature and conditions."
}

func (t *weatherTool) InputSchema() json.RawMessage {
	return tool.NewSchema().
		String("city", "The name of the city to get weather for", true).
		String("unit", "Temperature unit: celsius or fahrenheit", false).
		Build()
}

type weatherInput struct {
	City string `json:"city"`
	Unit string `json:"unit"`
}

func (t *weatherTool) Run(_ context.Context, raw json.RawMessage) (string, error) {
	var in weatherInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "", err
	}

	// Fake weather data for demo purposes
	weatherData := map[string]string{
		"tokyo":    `{"city": "Tokyo", "temperature": "18°C", "condition": "Partly Cloudy", "humidity": "65%"}`,
		"beijing":  `{"city": "Beijing", "temperature": "22°C", "condition": "Sunny", "humidity": "45%"}`,
		"shanghai": `{"city": "Shanghai", "temperature": "20°C", "condition": "Cloudy", "humidity": "55%"}`,
	}

	cityLower := strings.ToLower(in.City)
	if data, ok := weatherData[cityLower]; ok {
		return data, nil
	}
	return fmt.Sprintf(`{"city": "%s", "temperature": "20°C", "condition": "Unknown", "humidity": "50%%"}`, in.City), nil
}

// ─── Block printing ───────────────────────────────────────────────────────────

func printBlock(block *agent.StreamBlock) {
	switch block.Type {
	case agent.BlockThinkStart:
		fmt.Println("\n── 思考开始 ──────────────────")
	case agent.BlockThinkStream:
		fmt.Print(block.Content)
	case agent.BlockThinkEnd:
		fmt.Println("\n── 思考结束 ──────────────────")
		fmt.Printf("[完整内容]: %s\n", block.EndContent)
	case agent.BlockTextStart:
		fmt.Println("\n── 回答开始 ──────────────────")
	case agent.BlockTextStream:
		fmt.Print(block.Content)
	case agent.BlockTextEnd:
		fmt.Println("\n── 回答结束 ──────────────────")
		fmt.Printf("[完整内容]: %s\n", block.EndContent)
	case agent.BlockToolCall:
		fmt.Printf("\n── 工具调用 #%d ───────────────\n", block.Iteration)
		fmt.Printf("工具: %s\n参数: %s\n", block.ToolName, truncate(block.Content, 100))
	case agent.BlockToolResult:
		fmt.Printf("\n── 工具结果 #%d ───────────────\n", block.Iteration)
		if block.IsError {
			fmt.Printf("错误: %s\n", block.Content)
		} else {
			fmt.Printf("结果: %s\n", truncate(block.Content, 200))
		}
	case agent.BlockFinish:
		fmt.Printf("\n── 完成 #%d ───────────────────\n", block.Iteration)
		if block.TotalTokens > 0 {
			fmt.Printf("Token: 输入=%d 输出=%d 总计=%d\n",
				block.InputTokens, block.OutputTokens, block.TotalTokens)
		}
	case agent.BlockError:
		fmt.Printf("\n── 错误 #%d ───────────────────\n", block.Iteration)
		fmt.Printf("%s\n", block.Content)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
