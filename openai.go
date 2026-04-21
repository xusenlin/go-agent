// Package openai provides an OpenAI-compatible provider that works with
// OpenAI, DeepSeek, Moonshot, and any other OpenAI-spec compliant endpoint.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/xusenlin/go-agent/provider"
)

const defaultBaseURL = "https://api.openai.com/v1"
const defaultModel = "gpt-4o"

type Provider struct {
	apiKey  string
	baseURL string
	model   string
	http    *http.Client
}

type Option func(*Provider)

func WithModel(model string) Option   { return func(p *Provider) { p.model = model } }
func WithBaseURL(u string) Option     { return func(p *Provider) { p.baseURL = strings.TrimRight(u, "/") } }

func New(apiKey string, proxy *provider.ProxyConfig, opts ...Option) *Provider {
	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		model:   defaultModel,
		http:    proxy.HTTPClient(),
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string      { return "openai" }
func (p *Provider) Model() string     { return p.model }
func (p *Provider) SetModel(m string) { p.model = m }

// ─── Chat ─────────────────────────────────────────────────────────────────────

func (p *Provider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	body, err := p.buildBody(req, false)
	if err != nil {
		return nil, err
	}
	httpResp, err := p.do(ctx, body)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)

	var resp chatResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	return resp.toProviderResponse(), nil
}

// ─── Stream ───────────────────────────────────────────────────────────────────

func (p *Provider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	body, err := p.buildBody(req, true)
	if err != nil {
		return nil, err
	}
	httpResp, err := p.do(ctx, body)
	if err != nil {
		return nil, err
	}

	ch := make(chan *provider.Chunk, 64)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		var assembled provider.Response
		scanner := bufio.NewScanner(httpResp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}
			var ev streamEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				continue
			}
			for _, choice := range ev.Choices {
				delta := choice.Delta.Content
				assembled.Content += delta
				ch <- &provider.Chunk{Delta: delta}

				for _, tc := range choice.Delta.ToolCalls {
					// accumulate tool calls by index
					for len(assembled.ToolCalls) <= tc.Index {
						assembled.ToolCalls = append(assembled.ToolCalls, provider.ToolCall{})
					}
					atc := &assembled.ToolCalls[tc.Index]
					if tc.ID != "" {
						atc.ID = tc.ID
					}
					if tc.Function.Name != "" {
						atc.Name = tc.Function.Name
					}
					atc.Input = append(atc.Input, []byte(tc.Function.Arguments)...)
				}
				if choice.FinishReason == "tool_calls" {
					assembled.StopReason = "tool_use"
				} else if choice.FinishReason == "stop" {
					assembled.StopReason = "end_turn"
				}
			}
		}
		data, _ := json.Marshal(assembled)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

func (p *Provider) do(ctx context.Context, body []byte) (*http.Response, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	resp, err := p.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai: status %d: %s", resp.StatusCode, raw)
	}
	return resp, nil
}

func (p *Provider) buildBody(req *provider.Request, stream bool) ([]byte, error) {
	var msgs []map[string]any
	if req.System != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": req.System})
	}
	for _, m := range req.Messages {
		switch m.Role {
		case provider.RoleUser:
			if len(m.ToolResults) > 0 {
				for _, tr := range m.ToolResults {
					msgs = append(msgs, map[string]any{
						"role":         "tool",
						"tool_call_id": tr.ToolCallID,
						"content":      tr.Content,
					})
				}
			} else {
				msgs = append(msgs, map[string]any{"role": "user", "content": m.Content})
			}
		case provider.RoleAssistant:
			msg := map[string]any{"role": "assistant", "content": m.Content}
			if len(m.ToolCalls) > 0 {
				var tcs []map[string]any
				for _, tc := range m.ToolCalls {
					tcs = append(tcs, map[string]any{
						"id":   tc.ID,
						"type": "function",
						"function": map[string]any{
							"name":      tc.Name,
							"arguments": string(tc.Input),
						},
					})
				}
				msg["tool_calls"] = tcs
			}
			msgs = append(msgs, msg)
		}
	}

	var tools []map[string]any
	for _, t := range req.Tools {
		var schema any
		_ = json.Unmarshal(t.InputSchema, &schema)
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  schema,
			},
		})
	}

	payload := map[string]any{
		"model":    p.model,
		"messages": msgs,
		"stream":   stream,
	}
	if len(tools) > 0 {
		payload["tools"] = tools
	}
	return json.Marshal(payload)
}

// ─── JSON shapes ──────────────────────────────────────────────────────────────

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func (r *chatResponse) toProviderResponse() *provider.Response {
	if len(r.Choices) == 0 {
		return &provider.Response{}
	}
	c := r.Choices[0]
	resp := &provider.Response{Content: c.Message.Content, StopReason: "end_turn"}
	for _, tc := range c.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, provider.ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}
	if c.FinishReason == "tool_calls" {
		resp.StopReason = "tool_use"
	}
	return resp
}

type streamEvent struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}
