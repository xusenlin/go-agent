package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	ollamaapi "github.com/ollama/ollama/api"
	"github.com/xusenlin/go-agent/provider"
)

const defaultModel = "llama3.1"

type Provider struct {
	client *ollamaapi.Client
	model  string
}

type Option func(*Provider)

func WithModel(model string) Option { return func(p *Provider) { p.model = model } }

// New creates an Ollama provider.
// baseURL defaults to "http://localhost:11434" if empty.
func New(baseURL string, proxy *provider.ProxyConfig, opts ...Option) (*Provider, error) {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("ollama: invalid base url: %w", err)
	}
	httpClient := proxy.HTTPClient()
	if httpClient == http.DefaultClient {
		httpClient = &http.Client{}
	}
	p := &Provider{
		client: ollamaapi.NewClient(u, httpClient),
		model:  defaultModel,
	}
	for _, o := range opts {
		o(p)
	}
	return p, nil
}

func (p *Provider) Name() string      { return "ollama" }
func (p *Provider) Model() string     { return p.model }
func (p *Provider) SetModel(m string) { p.model = m }

func (p *Provider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	var assembled provider.Response
	streamCh, err := p.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	for chunk := range streamCh {
		if chunk.Done {
			// last chunk contains the JSON-encoded full response
			_ = json.Unmarshal([]byte(chunk.Delta), &assembled)
		}
	}
	return &assembled, nil
}

func (p *Provider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	ch := make(chan *provider.Chunk, 64)
	go func() {
		defer close(ch)

		var msgs []ollamaapi.Message
		if req.System != "" {
			msgs = append(msgs, ollamaapi.Message{Role: "system", Content: req.System})
		}
		for _, m := range req.Messages {
			role := string(m.Role)
			msgs = append(msgs, ollamaapi.Message{Role: role, Content: m.Content})
		}

		var tools []ollamaapi.Tool
		for _, t := range req.Tools {
			var params ollamaapi.ToolFunctionParams
			_ = json.Unmarshal(t.InputSchema, &params)
			tools = append(tools, ollamaapi.Tool{
				Type: "function",
				Function: ollamaapi.ToolFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  params,
				},
			})
		}

		oreq := &ollamaapi.ChatRequest{
			Model:    p.model,
			Messages: msgs,
			Tools:    tools,
			Stream:   boolPtr(true),
		}

		var assembled provider.Response
		err := p.client.Chat(ctx, oreq, func(resp ollamaapi.ChatResponse) error {
			delta := resp.Message.Content
			assembled.Content += delta
			ch <- &provider.Chunk{Delta: delta}

			if resp.Done {
				for _, tc := range resp.Message.ToolCalls {
					raw, _ := json.Marshal(tc.Function.Arguments)
					assembled.ToolCalls = append(assembled.ToolCalls, provider.ToolCall{
						ID:    tc.Function.Name,
						Name:  tc.Function.Name,
						Input: raw,
					})
				}
				assembled.StopReason = "end_turn"
				if len(assembled.ToolCalls) > 0 {
					assembled.StopReason = "tool_use"
				}
			}
			return nil
		})
		if err != nil {
			ch <- &provider.Chunk{Done: true}
			return
		}
		data, _ := json.Marshal(assembled)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

func boolPtr(b bool) *bool { return &b }
