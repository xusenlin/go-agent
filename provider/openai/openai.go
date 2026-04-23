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

const thinkStart = "<think>"
const thinkEnd = "</think>"

// Provider implements provider.Provider using OpenAI-compatible API.
type Provider struct {
	apiKey           string
	baseURL          string
	model            string
	http             *http.Client
	inThinking       bool // tracks whether we're inside a thinking block across chunks
	reasoningEffort  bool // whether to send reasoning_effort parameter
}

type Option func(*Provider)

func WithModel(model string) Option   { return func(p *Provider) { p.model = model } }
func WithBaseURL(u string) Option    { return func(p *Provider) { p.baseURL = strings.TrimRight(u, "/") } }
func WithReasoningEffort(enabled bool) Option { return func(p *Provider) { p.reasoningEffort = enabled } }

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

func (p *Provider) Name() string    { return "openai" }
func (p *Provider) Model() string   { return p.model }
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
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	var resp chatResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
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

				// Process delta - strip thinking tags and mark as IsThinking
				processed := p.processThinkingTags(delta)

				for _, seg := range processed {
					assembled.Content += seg.Text
					ch <- &provider.Chunk{Delta: seg.Text, IsThinking: seg.IsThinking}
				}

				// Accumulate tool calls
				for _, tc := range choice.Delta.ToolCalls {
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

			if ev.Usage != nil {
				data, _ := json.Marshal(assembled)
				ch <- &provider.Chunk{
					Done:          true,
					Delta:         string(data),
					InputTokens:   ev.Usage.PromptTokens,
					OutputTokens:  ev.Usage.CompletionTokens,
					TotalTokens:   ev.Usage.TotalTokens,
				}
				return
			}
		}
		data, _ := json.Marshal(assembled)
		ch <- &provider.Chunk{Done: true, Delta: string(data)}
	}()
	return ch, nil
}

// segment represents a processed text segment with its thinking status
type segment struct {
	Text       string
	IsThinking bool
}

// processThinkingTags parses delta text and returns segments with thinking status.
// It finds all <think> and </think> tags, splits text at these boundaries,
// and marks segments between <think> and </think> as thinking content.
// It maintains state across calls via p.inThinking to handle tags split across chunks.
func (p *Provider) processThinkingTags(delta string) []segment {
	if delta == "" {
		return nil
	}

	var result []segment
	start := 0

	for start < len(delta) {
		// Find next tag position
		nextStart := strings.Index(delta[start:], thinkStart)
		nextEnd := strings.Index(delta[start:], thinkEnd)

		// Determine which tag comes first
		var tagPos int
		var isStartTag bool

		switch {
		case nextStart == -1 && nextEnd == -1:
			// No more tags, emit remaining text
			result = append(result, segment{Text: delta[start:], IsThinking: p.inThinking})
			return result
		case nextStart == -1:
			tagPos = nextEnd
			isStartTag = false
		case nextEnd == -1:
			tagPos = nextStart
			isStartTag = true
		case nextStart < nextEnd:
			tagPos = nextStart
			isStartTag = true
		default:
			tagPos = nextEnd
			isStartTag = false
		}

		// Emit text before this tag
		if tagPos > 0 {
			result = append(result, segment{Text: delta[start : start+tagPos], IsThinking: p.inThinking})
		}

		// Skip past the tag
		if isStartTag {
			start += tagPos + len(thinkStart)
			p.inThinking = true
		} else {
			start += tagPos + len(thinkEnd)
			p.inThinking = false
		}
	}

	return result
}

func (p *Provider) do(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai: status %d [%s]: %s", resp.StatusCode, p.baseURL, raw)
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
	if stream {
		payload["stream_options"] = map[string]any{"include_usage": true}
	}

	// Add thinking level if specified and enabled
	if req.ThinkingLevel != "" && p.reasoningEffort {
		payload["reasoning_effort"] = string(req.ThinkingLevel)
	}

	return json.Marshal(payload)
}

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
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func (r *chatResponse) toProviderResponse() *provider.Response {
	if len(r.Choices) == 0 {
		return &provider.Response{}
	}
	c := r.Choices[0]
	resp := &provider.Response{Content: c.Message.Content, StopReason: "end_turn"}
	if r.Usage != nil {
		resp.Usage = provider.Usage{
			InputTokens:  r.Usage.PromptTokens,
			OutputTokens: r.Usage.CompletionTokens,
			TotalTokens:  r.Usage.TotalTokens,
		}
	}
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
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}
