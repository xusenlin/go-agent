package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"google.golang.org/genai"
	"github.com/xusenlin/go-agent/provider"
)

const defaultModel = "gemini-2.0-flash"

// Provider implements provider.Provider using the official Google GenAI SDK.
type Provider struct {
	client *genai.Client
	model  string
}

type Option func(*Provider)

func WithModel(model string) Option { return func(p *Provider) { p.model = model } }

func New(ctx context.Context, apiKey string, proxy *provider.ProxyConfig, opts ...Option) (*Provider, error) {
	cfg := &genai.ClientConfig{
		APIKey:     apiKey,
		HTTPClient: proxy.HTTPClient(),
	}
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("gemini: new client: %w", err)
	}
	p := &Provider{client: client, model: defaultModel}
	for _, o := range opts {
		o(p)
	}
	return p, nil
}

func (p *Provider) Name() string      { return "gemini" }
func (p *Provider) Model() string     { return p.model }
func (p *Provider) SetModel(m string) { p.model = m }

func (p *Provider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	model := p.client.GenerativeModel(p.model)
	p.applyTools(model, req.Tools)
	if req.System != "" {
		model.SystemInstruction = &genai.Content{Parts: []genai.Part{genai.Text(req.System)}}
	}

	session := model.StartChat()
	session.History = p.buildHistory(req.Messages[:len(req.Messages)-1])

	last := req.Messages[len(req.Messages)-1]
	resp, err := session.SendMessage(ctx, genai.Text(last.Content))
	if err != nil {
		return nil, fmt.Errorf("gemini chat: %w", err)
	}
	return p.convertResponse(resp), nil
}

func (p *Provider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	ch := make(chan *provider.Chunk, 64)
	go func() {
		defer close(ch)
		model := p.client.GenerativeModel(p.model)
		p.applyTools(model, req.Tools)
		if req.System != "" {
			model.SystemInstruction = &genai.Content{Parts: []genai.Part{genai.Text(req.System)}}
		}
		session := model.StartChat()
		session.History = p.buildHistory(req.Messages[:len(req.Messages)-1])
		last := req.Messages[len(req.Messages)-1]

		iter := session.SendMessageStream(ctx, genai.Text(last.Content))
		var assembled provider.Response
		for {
			resp, err := iter.Next()
			if err != nil {
				break
			}
			partial := p.convertResponse(resp)
			assembled.Content += partial.Content
			assembled.ToolCalls = append(assembled.ToolCalls, partial.ToolCalls...)
			assembled.StopReason = partial.StopReason
			ch <- &provider.Chunk{Delta: partial.Content}
		}
		data, _ := json.Marshal(assembled)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func (p *Provider) applyTools(model *genai.GenerativeModel, tools []provider.ToolDef) {
	if len(tools) == 0 {
		return
	}
	var fds []*genai.FunctionDeclaration
	for _, t := range tools {
		var schema genai.Schema
		_ = json.Unmarshal(t.InputSchema, &schema)
		fds = append(fds, &genai.FunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  &schema,
		})
	}
	model.Tools = []*genai.Tool{{FunctionDeclarations: fds}}
}

func (p *Provider) buildHistory(msgs []provider.Message) []*genai.Content {
	var history []*genai.Content
	for _, m := range msgs {
		role := "user"
		if m.Role == provider.RoleAssistant {
			role = "model"
		}
		history = append(history, &genai.Content{
			Role:  role,
			Parts: []genai.Part{genai.Text(m.Content)},
		})
	}
	return history
}

func (p *Provider) convertResponse(resp *genai.GenerateContentResponse) *provider.Response {
	r := &provider.Response{StopReason: "end_turn"}
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			switch v := part.(type) {
			case genai.Text:
				r.Content += string(v)
			case genai.FunctionCall:
				raw, _ := json.Marshal(v.Args)
				r.ToolCalls = append(r.ToolCalls, provider.ToolCall{
					ID:    v.Name, // Gemini uses name as ID
					Name:  v.Name,
					Input: raw,
				})
			}
		}
		if cand.FinishReason == genai.FinishReasonStop && len(r.ToolCalls) > 0 {
			r.StopReason = "tool_use"
		}
	}
	return r
}
