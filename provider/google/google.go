package google

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/xusenlin/go-agent/provider"
	"google.golang.org/genai"
)

const defaultModel = "gemini-2.0-flash"

// Provider implements provider.Provider using the official Google GenAI SDK.
type Provider struct {
	apiKey  string
	baseURL string
	proxy   *provider.ProxyConfig
	client  *genai.Client
	model   string
}

// Option is a functional option setter.
type Option func(*Provider)

func WithModel(model string) Option   { return func(p *Provider) { p.model = model } }
func WithBaseURL(u string) Option     { return func(p *Provider) { p.baseURL = strings.TrimRight(u, "/") } }

func New(apiKey string, proxy *provider.ProxyConfig, opts ...Option) *Provider {
	p := &Provider{
		apiKey: apiKey,
		proxy:  proxy,
		model:  defaultModel,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) ensureClient() error {
	if p.client != nil {
		return nil
	}
	cfg := &genai.ClientConfig{
		APIKey:     p.apiKey,
		HTTPClient: p.proxy.HTTPClient(),
	}
	if p.baseURL != "" {
		cfg.HTTPOptions.BaseURL = p.baseURL
	}
	client, err := genai.NewClient(context.Background(), cfg)
	if err != nil {
		return fmt.Errorf("google: new client: %w", err)
	}
	p.client = client
	return nil
}

func (p *Provider) Name() string      { return "google" }
func (p *Provider) Model() string     { return p.model }
func (p *Provider) SetModel(m string) { p.model = m }

func (p *Provider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	if err := p.ensureClient(); err != nil {
		return nil, err
	}
	contents := p.buildContents(req.Messages)
	tools := p.buildTools(req.Tools)
	cfg := &genai.GenerateContentConfig{
		Tools:             tools,
		SystemInstruction: p.buildSystem(req.System),
	}

	// Add thinking level if specified
	if req.ThinkingLevel != "" {
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
		}
	}

	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents, cfg)
	if err != nil {
		return nil, fmt.Errorf("google chat: %w", err)
	}
	return p.convertResponse(resp), nil
}

func (p *Provider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	if err := p.ensureClient(); err != nil {
		return nil, err
	}
	ch := make(chan *provider.Chunk, 64)
	go func() {
		defer close(ch)

		contents := p.buildContents(req.Messages)
		tools := p.buildTools(req.Tools)
		cfg := &genai.GenerateContentConfig{
			Tools:             tools,
			SystemInstruction: p.buildSystem(req.System),
		}

		// Add thinking level if specified
		if req.ThinkingLevel != "" {
			cfg.ThinkingConfig = &genai.ThinkingConfig{
				IncludeThoughts: true,
			}
		}

		var assembled provider.Response
		for resp, err := range p.client.Models.GenerateContentStream(ctx, p.model, contents, cfg) {
			if err != nil {
				ch <- &provider.Chunk{Done: true}
				return
			}
			partial := p.convertResponse(resp)
			if partial != nil {
				// Handle thinking content
				if partial.Thinking != "" {
					ch <- &provider.Chunk{Delta: partial.Thinking, IsThinking: true}
				}
				// Handle regular content
				if partial.Content != "" {
					ch <- &provider.Chunk{Delta: partial.Content}
				}
				assembled.Content += partial.Content
				assembled.Thinking += partial.Thinking
				assembled.ToolCalls = append(assembled.ToolCalls, partial.ToolCalls...)
				assembled.StopReason = partial.StopReason
			}
		}
		data, err := json.Marshal(assembled)
		if err != nil {
			ch <- &provider.Chunk{Done: true}
			return
		}
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func (p *Provider) buildSystem(system string) *genai.Content {
	if system == "" {
		return nil
	}
	return &genai.Content{
		Parts: []*genai.Part{{Text: system}},
	}
}

func (p *Provider) buildContents(msgs []provider.Message) []*genai.Content {
	contents := make([]*genai.Content, 0, len(msgs))
	for _, m := range msgs {
		role := "user"
		if m.Role == provider.RoleAssistant {
			role = "model"
		}
		var parts []*genai.Part
		if m.Content != "" {
			parts = append(parts, &genai.Part{Text: m.Content})
		}
		for _, tc := range m.ToolCalls {
			input, _ := json.Marshal(tc.Input)
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.ID,
					Name: tc.Name,
					Args: unmarshalMap(input),
				},
			})
		}
		for _, tr := range m.ToolResults {
			content := tr.Content
			if tr.IsError {
				content = "Error: " + content
			}
			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       tr.ToolCallID,
					Name:     "", // Gemini doesn't echo function name in response
					Response: map[string]any{"result": content},
				},
			})
		}
		if len(parts) > 0 {
			contents = append(contents, &genai.Content{Role: role, Parts: parts})
		}
	}
	return contents
}

func (p *Provider) buildTools(defs []provider.ToolDef) []*genai.Tool {
	if len(defs) == 0 {
		return nil
	}
	tools := make([]*genai.Tool, 0, len(defs))
	for _, t := range defs {
		var schema *genai.Schema
		if len(t.InputSchema) > 0 {
			_ = json.Unmarshal(t.InputSchema, &schema)
		}
		tools = append(tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  schema,
			}},
		})
	}
	return tools
}

func (p *Provider) convertResponse(resp *genai.GenerateContentResponse) *provider.Response {
	r := &provider.Response{StopReason: "end_turn"}
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			switch {
			case part.Text != "" && part.Thought:
				// This is thinking content
				r.Thinking += part.Text
			case part.Text != "":
				r.Content += part.Text
			case part.FunctionCall != nil:
				args, _ := json.Marshal(part.FunctionCall.Args)
				id := part.FunctionCall.ID
				if id == "" {
					id = fmt.Sprintf("call_%s", part.FunctionCall.Name)
				}
				r.ToolCalls = append(r.ToolCalls, provider.ToolCall{
					ID:    id,
					Name:  part.FunctionCall.Name,
					Input: args,
				})
			}
		}
		if cand.FinishReason == genai.FinishReasonStop && len(r.ToolCalls) > 0 {
			r.StopReason = "tool_use"
		}
	}
	return r
}

func unmarshalMap(data json.RawMessage) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return make(map[string]any)
	}
	return m
}
