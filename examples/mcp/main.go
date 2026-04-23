package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/xusenlin/go-agent/agent"
	"github.com/xusenlin/go-agent/config"
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

	a, err := agent.New().
		WithProvider(openai.New(apiKey, nil, openai.WithBaseURL(baseURL))).
		WithModel(model).
		WithMCP("npx @modelcontextprotocol/server-filesystem /tmp").
		WithMaxIter(5).
		Build()
	if err != nil {
		log.Fatalf("build error: %v", err)
	}

	// ── Run with streaming ─────────────────────────────────────────────────────
	query := "帮我看看tmp目录有什么文件"
	fmt.Println("Query:", query)
	fmt.Println(strings.Repeat("─", 60))

	blocks, err := a.RunStream(ctx, query)
	if err != nil {
		log.Fatalf("run error: %v", err)
	}

	// ── Process streamed blocks ─────────────────────────────────────────────────
	for block := range blocks {
		printBlock2(block)
	}

	fmt.Println(strings.Repeat("─", 60))
	fmt.Println("Stream completed.")
}

// ─── Weather tool ──────────────────────────────────────────────────────────────

// ─── Block printing ───────────────────────────────────────────────────────────

func printBlock2(block *agent.StreamBlock) {
	switch block.Type {
	case agent.BlockThinkStart:
		fmt.Println("\n── 思考开始 ──────────────────")
	case agent.BlockThinkStream:
		fmt.Print(block.Delta)
	case agent.BlockThinkEnd:
		fmt.Println("\n── 思考结束 ──────────────────")
		fmt.Printf("[完整内容]: %s\n", block.Full)
	case agent.BlockTextStart:
		fmt.Println("\n── 回答开始 ──────────────────")
	case agent.BlockTextStream:
		fmt.Print(block.Delta)
	case agent.BlockTextEnd:
		fmt.Println("\n── 回答结束 ──────────────────")
		fmt.Printf("[完整内容]: %s\n", block.Full)
	case agent.BlockToolCall:
		fmt.Printf("\n── 工具调用 #%d ───────────────\n", block.Iteration)
		fmt.Printf("工具: %s\n参数: %s\n", block.ToolName, truncate(block.Payload, 100))
	case agent.BlockToolResult:
		fmt.Printf("\n── 工具结果 #%d ───────────────\n", block.Iteration)
		if block.IsError {
			fmt.Printf("错误: %s\n", block.Payload)
		} else {
			fmt.Printf("结果: %s\n", truncate(block.Payload, 200))
		}
	case agent.BlockFinish:
		fmt.Printf("\n── 完成 #%d ───────────────────\n", block.Iteration)
		if block.TotalTokens > 0 {
			fmt.Printf("Token: 输入=%d 输出=%d 总计=%d\n",
				block.InputTokens, block.OutputTokens, block.TotalTokens)
		}
	case agent.BlockError:
		fmt.Printf("\n── 错误 #%d ───────────────────\n", block.Iteration)
		fmt.Printf("%s\n", block.Full)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
