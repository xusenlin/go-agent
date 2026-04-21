package anthropic

import (
	"context"
	"encoding/json"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/xusenlin/go-agent/provider"
)

const defaultModel = "claude-opus-4-7"

// Provider implements provider.Provider using the official Anthropic SDK.
type Provider struct {
	client *sdk.Client
	model  string
}

// Option configures the Anthropic provider.
type Option func(*Provider)

func WithModel(model string) Option { return func(p *Provider) { p.model = model } }

// New creates an Anthropic provider.
// proxy is optional; pass nil to use the default http.Client.
func New(apiKey string, proxy *provider.ProxyConfig, opts ...Option) *Provider {
	clientOpts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if proxy != nil {
		clientOpts = append(clientOpts, option.WithHTTPClient(proxy.HTTPClient()))
	}
	p := &Provider{
		client: sdk.NewClient(clientOpts...),
		model:  defaultModel,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string          { return "anthropic" }
func (p *Provider) Model() string         { return p.model }
func (p *Provider) SetModel(m string)     { p.model = m }

func (p *Provider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	params, err := p.buildParams(req)
	if err != nil {
		return nil, err
	}
	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic chat: %w", err)
	}
	return p.convertResponse(msg), nil
}

func (p *Provider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	params, err := p.buildParams(req)
	if err != nil {
		return nil, err
	}
	stream := p.client.Messages.NewStreaming(ctx, params)
	ch := make(chan *provider.Chunk, 64)

	go func() {
		defer close(ch)
		for stream.Next() {
			ev := stream.Current()
			switch v := ev.AsUnion().(type) {
			case sdk.ContentBlockDeltaEvent:
				switch d := v.Delta.AsUnion().(type) {
				case sdk.TextDelta:
					ch <- &provider.Chunk{Delta: d.Text}
				case sdk.InputJSONDelta:
					ch <- &provider.Chunk{InputDelta: d.PartialJSON}
				}
			case sdk.MessageStreamEvent:
				_ = v
			}
		}
		if err := stream.Err(); err != nil {
			ch <- &provider.Chunk{Done: true}
			return
		}
		// emit final assembled message
		final := p.convertResponse(stream.GetFinalMessage())
		data, _ := json.Marshal(final)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()

	return ch, nil
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func (p *Provider) buildParams(req *provider.Request) (sdk.MessageNewParams, error) {
	var msgs []sdk.MessageParam
	for _, m := range req.Messages {
		switch m.Role {
		case provider.RoleUser:
			if len(m.ToolResults) > 0 {
				var blocks []sdk.ContentBlockParamUnion
				for _, tr := range m.ToolResults {
					blocks = append(blocks, sdk.NewToolResultBlock(tr.ToolCallID, tr.Content, tr.IsError))
				}
				msgs = append(msgs, sdk.NewUserMessage(blocks...))
			} else {
				msgs = append(msgs, sdk.NewUserMessage(sdk.NewTextBlock(m.Content)))
			}
		case provider.RoleAssistant:
			var blocks []sdk.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, sdk.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var input interface{}
				_ = json.Unmarshal(tc.Input, &input)
				blocks = append(blocks, sdk.NewToolUseBlock(tc.ID, tc.Name, input))
			}
			msgs = append(msgs, sdk.NewAssistantMessage(blocks...))
		}
	}

	params := sdk.MessageNewParams{
		Model:     sdk.Model(p.model),
		MaxTokens: 8192,
		Messages:  msgs,
	}
	if req.System != "" {
		params.System = []sdk.TextBlockParam{{Text: req.System}}
	}
	for _, t := range req.Tools {
		var schema interface{}
		_ = json.Unmarshal(t.InputSchema, &schema)
		params.Tools = append(params.Tools, sdk.ToolParam{
			Name:        t.Name,
			Description: sdk.String(t.Description),
			InputSchema: sdk.ToolInputSchemaParam{Properties: schema},
		})
	}
	return params, nil
}

func (p *Provider) convertResponse(msg *sdk.Message) *provider.Response {
	resp := &provider.Response{StopReason: string(msg.StopReason)}
	for _, block := range msg.Content {
		switch v := block.AsUnion().(type) {
		case sdk.TextBlock:
			resp.Content += v.Text
		case sdk.ToolUseBlock:
			raw, _ := json.Marshal(v.Input)
			resp.ToolCalls = append(resp.ToolCalls, provider.ToolCall{
				ID:    v.ID,
				Name:  v.Name,
				Input: raw,
			})
		}
	}
	if len(resp.ToolCalls) > 0 {
		resp.StopReason = "tool_use"
	}
	return resp
}
