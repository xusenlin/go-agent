package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/xusenlin/go-agent/provider"
)

const defaultModel = "claude-opus-4-7"
const defaultBaseURL = "https://api.anthropic.com/v1/messages"

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

func (p *Provider) Name() string      { return "anthropic" }
func (p *Provider) Model() string     { return p.model }
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

	raw, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: read response: %w", err)
	}
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

				// Check if response is HTML (not JSON)
				if len(buf) > 0 && (buf[0] == '<' || bytes.HasPrefix(bytes.TrimSpace(buf), []byte("<!doctype"))) {
					ch <- &provider.Chunk{
						Error: fmt.Sprintf("API returned HTML instead of JSON. Please check baseURL (current: %s)", p.baseURL),
					}
					return
				}

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
					case "content_block_start":
						log.Printf("[ANTHROPIC] content_block_start: %+v\n", event)
					case "content_block_delta":
						log.Printf("[ANTHROPIC] content_block_delta: type=%s, text=%q, thinking=%q\n", 
							event.Delta.Type, event.Delta.Text, event.Delta.Thinking)
						if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
							assembled.Content += event.Delta.Text
							ch <- &provider.Chunk{Delta: event.Delta.Text}
						} else if event.Delta.Type == "thinking_delta" && event.Delta.Thinking != "" {
							ch <- &provider.Chunk{Delta: event.Delta.Thinking, IsThinking: true}
						}
					case "message_delta":
						if event.StopReason != "" {
							assembled.StopReason = event.StopReason
						}
						// Send usage stats from message_delta (Anthropic sends output_tokens here)
						if event.Usage.OutputTokens > 0 {
							ch <- &provider.Chunk{
								InputTokens:  event.Usage.InputTokens,
								OutputTokens: event.Usage.OutputTokens,
								TotalTokens:  event.Usage.InputTokens + event.Usage.OutputTokens,
							}
						}
					case "message_stop":
						// Anthropic sends final message_stop
						ch <- &provider.Chunk{
							Done:         true,
							InputTokens:  event.Usage.InputTokens,
							OutputTokens: event.Usage.OutputTokens,
							TotalTokens:  event.Usage.InputTokens + event.Usage.OutputTokens,
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
	for {
		// Find the next newline
		idx := bytes.IndexByte(data, '\n')
		if idx < 0 {
			// No newline found, check if data starts with "data: "
			if bytes.HasPrefix(data, []byte("data: ")) {
				event := string(data[6:])
				if event == "[DONE]" {
					return "", nil, false
				}
				return event, nil, true
			}
			return "", data, false
		}

		line := data[:idx]
		remaining := data[idx+1:]

		if bytes.HasPrefix(line, []byte("data: ")) {
			event := string(line[6:])
			if event == "[DONE]" {
				return "", remaining, false
			}
			return event, remaining, true
		}

		data = remaining
	}
}

func (p *Provider) do(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create request: %w", err)
	}
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
		return nil, fmt.Errorf("anthropic: status %d [%s]: %s", resp.StatusCode, p.baseURL, raw)
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
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": json.RawMessage(tc.Input),
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
		"model":      p.model,
		"max_tokens": 8192,
		"messages":   messages,
		"stream":     stream,
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
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": schema,
			})
		}
		payload["tools"] = tools
	}

	// Add thinking level if specified
	if req.ThinkingLevel != "" {
		budgetTokens := 0
		switch req.ThinkingLevel {
		case provider.ThinkingLevelLow:
			budgetTokens = 2000
		case provider.ThinkingLevelMedium:
			budgetTokens = 5000
		case provider.ThinkingLevelHigh:
			budgetTokens = 10000
		}

		if budgetTokens > 0 {
			payload["thinking"] = map[string]any{
				"type":          "enabled",
				"budget_tokens": budgetTokens,
			}
		}
	}

	return json.Marshal(payload)
}

type anthropicResponse struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type  string `json:"type"`
		Text  string `json:"text"`
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`
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
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		Thinking string `json:"thinking"` // for thinking_delta
		Part     string `json:"partial_json"`
	} `json:"delta"`
	StopReason string    `json:"stop_reason"`
	Usage      usageInfo `json:"usage"`
}

type usageInfo struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}
