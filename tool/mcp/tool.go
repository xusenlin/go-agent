package mcp

import (
	"context"
	"encoding/json"

	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/tool"
)

// mcpTool adapts a single MCP remote tool to the tool.Tool interface.
type mcpTool struct {
	info   ToolInfo
	client *Client
}

func (t *mcpTool) Name() string                 { return t.info.Name }
func (t *mcpTool) Description() string          { return t.info.Description }
func (t *mcpTool) InputSchema() json.RawMessage { return t.info.InputSchema }

func (t *mcpTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	return t.client.Call(ctx, t.info.Name, input)
}

// DiscoverTools connects to an MCP server, discovers its tools, and returns
// them as []tool.Tool ready to be passed to agent.Builder.WithTools().
func DiscoverTools(ctx context.Context, spec string, proxy *provider.ProxyConfig) ([]tool.Tool, *Client, error) {
	client, err := NewClient(spec, proxy)
	if err != nil {
		return nil, nil, err
	}
	infos, err := client.Discover(ctx)
	if err != nil {
		_ = client.Close()
		return nil, nil, err
	}
	tools := make([]tool.Tool, len(infos))
	for i, info := range infos {
		tools[i] = &mcpTool{info: info, client: client}
	}
	return tools, client, nil
}