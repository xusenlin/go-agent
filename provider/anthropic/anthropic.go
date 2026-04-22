package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/xusenlin/go-agent/provider"
)

const defaultModel = "claude-opus-4-7"
const baseURL = "https://api.anthropic.com/v1/messages"

type Provider struct {
	apiKey string
	model  string
	http   *http.Client
}

func New(apiKey string, proxy *provider.ProxyConfig) *Provider {
	p := &Provider{
		apiKey: apiKey,
		model:  defaultModel,
		http:   proxy.HTTPClient(),
	}
	return p
}

func (p *Provider) Name() string  { return "anthropic" }
func (p *Provider) Model() string { return p.model }
func (p *Provider) SetModel(m string) { p.model = m }

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
	var resp anthropicResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("anthropic: parse response: %w", err)
	}

	return resp.toProviderResponse(), nil
}

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
		reader := httpResp.Body
		buf := make([]byte, 0, 1024)
		tmp := make([]byte, 1024)

		for {
			n, err := reader.Read(tmp)
			if n > 0 {
				buf = append(buf, tmp[:n]...)
				// Process complete events
				for {
					data, remaining, found := extractEvent(buf)
					if !found {
						break
					}
					buf = remaining

					var event streamEvent
					if err := json.Unmarshal([]byte(data), &event); err != nil {
						continue
					}

					switch event.Type {
					case "content_block_delta":
						if event.Delta.Type == "text_delta" {
							assembled.Content += event.Delta.Text
							ch <- &provider.Chunk{Delta: event.Delta.Text}
						}
					case "message_delta":
						if event.StopReason != "" {
							assembled.StopReason = event.StopReason
						}
					}
				}
			}
			if err != nil {
				break
			}
		}

		data, _ := json.Marshal(assembled)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

func extractEvent(data []byte) (string, []byte, bool) {
	lines := strings.Split(string(data), "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			event := strings.TrimPrefix(line, "data: ")
			if event == "[DONE]" {
				continue
			}
			remaining := []byte(strings.Join(lines[i+1:], "\n"))
			return event, remaining, true
		}
	}
	return "", data, false
}

func (p *Provider) do(ctx context.Context, body []byte) (*http.Response, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, baseURL, strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := p.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic: status %d: %s", resp.StatusCode, raw)
	}
	return resp, nil
}

func (p *Provider) buildBody(req *provider.Request, stream bool) ([]byte, error) {
	messages := make([]map[string]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		switch m.Role {
		case provider.RoleUser:
			if len(m.ToolResults) > 0 {
				for _, tr := range m.ToolResults {
					content := tr.Content
					if tr.IsError {
						content = "Error: " + content
					}
					messages = append(messages, map[string]any{
						"role": "user",
						"content": []map[string]any{
							{"type": "tool_result", "tool_use_id": tr.ToolCallID, "content": content},
						},
					})
				}
			} else {
				messages = append(messages, map[string]any{"role": "user", "content": m.Content})
			}
		case provider.RoleAssistant:
			content := m.Content
			var toolUses []map[string]any
			for _, tc := range m.ToolCalls {
				toolUses = append(toolUses, map[string]any{
					"type":    "tool_use",
					"id":      tc.ID,
					"name":    tc.Name,
					"input":   json.RawMessage(tc.Input),
				})
			}
			blocks := make([]any, 0)
			if content != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": content})
			}
			for _, tu := range toolUses {
				blocks = append(blocks, tu)
			}
			messages = append(messages, map[string]any{"role": "assistant", "content": blocks})
		}
	}

	payload := map[string]any{
		"model":       p.model,
		"max_tokens":  8192,
		"messages":    messages,
		"stream":      stream,
	}

	if req.System != "" {
		payload["system"] = req.System
	}

	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			var schema any
			_ = json.Unmarshal(t.InputSchema, &schema)
			tools = append(tools, map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"input_schema": schema,
			})
		}
		payload["tools"] = tools
	}

	return json.Marshal(payload)
}

type anthropicResponse struct {
	Type        string `json:"type"`
	Role        string `json:"role"`
	Content     []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		ID      string `json:"id"`
		Name    string `json:"name"`
		Input   any    `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
}

func (r *anthropicResponse) toProviderResponse() *provider.Response {
	resp := &provider.Response{StopReason: r.StopReason}
	for _, block := range r.Content {
		switch block.Type {
		case "text":
			resp.Content += block.Text
		case "tool_use":
			input, _ := json.Marshal(block.Input)
			resp.ToolCalls = append(resp.ToolCalls, provider.ToolCall{
				ID:    block.ID,
				Name:  block.Name,
				Input: input,
			})
		}
	}
	if len(resp.ToolCalls) > 0 {
		resp.StopReason = "tool_use"
	}
	return resp
}

type streamEvent struct {
	Type       string `json:"type"`
	Index      int    `json:"index"`
	Delta      struct {
		Type   string `json:"type"`
		Text   string `json:"text"`
		Part   string `json:"partial_json"`
	} `json:"delta"`
	StopReason string `json:"stop_reason"`
}