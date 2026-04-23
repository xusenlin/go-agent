package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/xusenlin/go-agent/hook"
	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/tool"
)

// BlockType categorises each unit of streamed output.
type BlockType string

const (
	BlockThinkStart  BlockType = "think_start"  // thinking block started (no content)
	BlockThinkStream BlockType = "think_stream" // thinking content chunk (incremental)
	BlockThinkEnd    BlockType = "think_end"    // thinking block ended (EndContent has full text)
	BlockTextStart   BlockType = "text_start"   // text block started (no content)
	BlockTextStream  BlockType = "text_stream"  // text content chunk (incremental)
	BlockTextEnd     BlockType = "text_end"     // text block ended (EndContent has full text)
	BlockToolCall    BlockType = "tool_call"    // tool invocation request
	BlockToolResult  BlockType = "tool_result"  // tool output (success or error)
	BlockError       BlockType = "error"        // fatal error, stream aborts
	BlockFinish      BlockType = "finish"       // agent completed successfully
)

// StreamBlock is one atomic unit emitted by RunStream.
// Every block carries a token delta so callers can persist usage metadata.
//
// Content fields (mutually exclusive, determined by Type):
//   - Delta:   streaming increment (ThinkStream, TextStream)
//   - Full:    complete content (ThinkEnd, TextEnd, Error)
//   - Payload: event data (ToolCall input, ToolResult output)
type StreamBlock struct {
	Type         BlockType `json:"type"`
	Iteration    int       `json:"iteration"`
	Delta        string    `json:"delta,omitempty"`    // streaming increment: ThinkStream, TextStream
	Full         string    `json:"full,omitempty"`     // complete content: ThinkEnd, TextEnd, Error
	Payload      string    `json:"payload,omitempty"`  // event data: ToolCall input, ToolResult output
	ToolName     string    `json:"tool_name,omitempty"`
	ToolID       string    `json:"tool_id,omitempty"`
	IsError      bool      `json:"is_error,omitempty"`
	InputTokens  int       `json:"input_tokens,omitempty"`
	OutputTokens int       `json:"output_tokens,omitempty"`
	TotalTokens  int       `json:"total_tokens,omitempty"`
}

// RunStream is the streaming counterpart of Run.
// It drives the ReAct loop exactly like Run but pushes every meaningful
// event through the returned channel as a StreamBlock so callers can
// persist tokens, render UI, or write to a database in real time.
//
// The channel is closed when the stream finishes or a fatal error occurs.
// RunStream always calls the hook methods (OnThinkStart, OnToolStart, …)
// in addition to emitting blocks.
func (a *Agent) RunStream(ctx context.Context, input string) (<-chan *StreamBlock, error) {
	blocks := make(chan *StreamBlock, 64)

	toolDefs := tool.ToProviderDefs(a.registry.All())

	messages := []provider.Message{
		{Role: provider.RoleUser, Content: input},
	}

	// ── OnAgentStart ─────────────────────────────────────────────────────────
	if err := a.hooks.OnAgentStart(ctx, &hook.AgentStartEvent{Input: input}); err != nil {
		blocks <- &StreamBlock{Type: BlockError, Full: fmt.Sprintf("OnAgentStart: %v", err)}
		close(blocks)
		return blocks, err
	}

	// ── ReAct loop ────────────────────────────────────────────────────────────
	go func() {
		defer close(blocks)

		for iter := 1; iter <= a.maxIter; iter++ {
			req := &provider.Request{
				System:        a.systemPrompt,
				Messages:      messages,
				Tools:         toolDefs,
				Stream:        true,
				ThinkingLevel: a.thinkingLevel,
			}

			// ── Think ─────────────────────────────────────────────────────────
			if err := a.hooks.OnThinkStart(ctx, &hook.ThinkStartEvent{
				Iteration: iter,
				Messages:  messages,
			}); err != nil {
				blocks <- &StreamBlock{Type: BlockError, Full: fmt.Sprintf("OnThinkStart: %v", err)}
				return
			}

			resp, tokens, err := a.streamAndAssembleStream(ctx, req, iter, blocks)
			if err != nil {
				a.emitError(blocks, iter, err)
				_ = a.hooks.OnAgentError(ctx, &hook.AgentErrorEvent{Err: err, Iterations: iter})
				return
			}

			if err := a.hooks.OnThinkEnd(ctx, &hook.ThinkEndEvent{
				Iteration:    iter,
				Response:     resp,
				StreamTokens: tokens.content,
			}); err != nil {
				blocks <- &StreamBlock{Type: BlockError, Full: fmt.Sprintf("OnThinkEnd: %v", err)}
				return
			}

			// ── End turn: no more tool calls ───────────────────────────────────
			if resp.StopReason == "end_turn" || len(resp.ToolCalls) == 0 {
				output := resp.Content
				blocks <- &StreamBlock{
					Type:         BlockFinish,
					Iteration:    iter,
					InputTokens:  tokens.input,
					OutputTokens: tokens.output,
					TotalTokens:  tokens.total,
				}
				_ = a.hooks.OnAgentFinish(ctx, &hook.AgentFinishEvent{
					Output:     output,
					Iterations: iter,
				})
				return
			}

			// ── Append assistant message (with tool calls) to history ──────────
			messages = append(messages, provider.Message{
				Role:      provider.RoleAssistant,
				Content:   resp.Content,
				ToolCalls: resp.ToolCalls,
			})

			// ── Execute tool calls ─────────────────────────────────────────────
			for _, tc := range resp.ToolCalls {
				result := a.executeToolStream(ctx, iter, tc, blocks)
				messages = append(messages, provider.Message{
					Role:        provider.RoleUser,
					ToolResults: []provider.ToolResult{result},
				})
			}
		}

		// ── Max iterations exceeded ─────────────────────────────────────────
		err := fmt.Errorf("agent: max iterations (%d) exceeded", a.maxIter)
		blocks <- &StreamBlock{Type: BlockError, Iteration: a.maxIter, Full: err.Error()}
		_ = a.hooks.OnAgentError(ctx, &hook.AgentErrorEvent{Err: err, Iterations: a.maxIter})
	}()

	return blocks, nil
}

// tokenSnapshot holds cumulative token counts from a streaming response.
type tokenSnapshot struct {
	content              string
	input, output, total int
}

// streamAndAssembleStream streams tokens to the block channel and returns
// the fully assembled Response when done.
func (a *Agent) streamAndAssembleStream(ctx context.Context, req *provider.Request, iter int, blocks chan<- *StreamBlock) (*provider.Response, *tokenSnapshot, error) {
	streamCh, err := a.provider.Stream(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	var (
		assembled     provider.Response
		tokens        tokenSnapshot
		thinkBuilder  strings.Builder
		textBuilder   strings.Builder
		lastIsThink   *bool // nil = no previous chunk
		lastInputTok  int
		lastOutputTok int
		lastTotalTok  int
		textStarted   bool // track if BlockTextStart was sent
	)

	flushThink := func() {
		content := thinkBuilder.String()
		if content != "" {
			blocks <- &StreamBlock{
				Type:         BlockThinkEnd,
				Iteration:    iter,
				Full:         content,
				InputTokens:  lastInputTok,
				OutputTokens: lastOutputTok,
				TotalTokens:  lastTotalTok,
			}
			thinkBuilder.Reset()
		}
	}

	flushText := func() {
		if !textStarted {
			return
		}
		content := textBuilder.String()
		blocks <- &StreamBlock{
			Type:         BlockTextEnd,
			Iteration:    iter,
			Full:         strings.TrimSpace(content),
			InputTokens:  lastInputTok,
			OutputTokens: lastOutputTok,
			TotalTokens:  lastTotalTok,
		}
		textBuilder.Reset()
		textStarted = false
	}

	for chunk := range streamCh {
		// Check for error from provider
		if chunk.Error != "" {
			return nil, nil, fmt.Errorf("%s", chunk.Error)
		}

		// Accumulate usage stats from any chunk that has them
		if chunk.InputTokens > 0 || chunk.OutputTokens > 0 || chunk.TotalTokens > 0 {
			tokens.input = chunk.InputTokens
			tokens.output = chunk.OutputTokens
			tokens.total = chunk.TotalTokens
			lastInputTok = chunk.InputTokens
			lastOutputTok = chunk.OutputTokens
			lastTotalTok = chunk.TotalTokens
		}

		if chunk.Done {
			// Final chunk - unmarshal Delta JSON to extract ToolCalls and StopReason
			if err := json.Unmarshal([]byte(chunk.Delta), &assembled); err != nil {
				log.Printf("[AGENT] Failed to unmarshal Done chunk: %v", err)
			}
			continue
		}

		// Check if thinking state changed
		if lastIsThink == nil || *lastIsThink != chunk.IsThinking {
			// Flush previous block if switching state (skip initial nil state)
			if lastIsThink != nil {
				if *lastIsThink {
					flushThink()
				} else {
					flushText()
				}
			}
			lastIsThink = &chunk.IsThinking

			// Send think start event immediately
			if chunk.IsThinking {
				blocks <- &StreamBlock{Type: BlockThinkStart, Iteration: iter}
			}
			// Text start is delayed until we have content
		}

		if chunk.IsThinking {
			thinkBuilder.WriteString(chunk.Delta)
			blocks <- &StreamBlock{
				Type:      BlockThinkStream,
				Iteration: iter,
				Delta:     chunk.Delta,
			}
		} else {
			// Send text start on first actual non-empty content
			if !textStarted && strings.TrimSpace(chunk.Delta) != "" {
				blocks <- &StreamBlock{Type: BlockTextStart, Iteration: iter}
				textStarted = true
			}
			if textStarted {
				textBuilder.WriteString(chunk.Delta)
				blocks <- &StreamBlock{
					Type:      BlockTextStream,
					Iteration: iter,
					Delta:     chunk.Delta,
				}
			}
		}
	}

	// Flush any remaining content
	if lastIsThink != nil {
		flushThink()
		flushText()
	}

	tokens.content = textBuilder.String()
	return &assembled, &tokens, nil
}

// executeToolStream executes a single tool call and emits blocks for the call
// and its result. Returns the ToolResult so it can be added to message history.
func (a *Agent) executeToolStream(ctx context.Context, iter int, tc provider.ToolCall, blocks chan<- *StreamBlock) provider.ToolResult {
	// ── OnToolStart ──────────────────────────────────────────────────────────
	startEvent := &hook.ToolStartEvent{
		Iteration: iter,
		ToolName:  tc.Name,
		CallID:    tc.ID,
		Input:     tc.Input,
	}
	if err := a.hooks.OnToolStart(ctx, startEvent); err != nil {
		result := errorResult(tc.ID, fmt.Errorf("hook OnToolStart: %w", err))
		blocks <- &StreamBlock{
			Type:      BlockError,
			Iteration: iter,
			Full:      result.Content,
		}
		return result
	}

	// Emit tool call block
	blocks <- &StreamBlock{
		Type:      BlockToolCall,
		Iteration: iter,
		ToolName:  tc.Name,
		ToolID:    tc.ID,
		Payload:   string(startEvent.Input),
	}

	// ── Look up and run tool ─────────────────────────────────────────────────
	t, ok := a.registry.Get(tc.Name)
	if !ok {
		runErr := fmt.Errorf("unknown tool: %q", tc.Name)
		endEvent := &hook.ToolEndEvent{
			Iteration: iter, ToolName: tc.Name, CallID: tc.ID,
			Output: runErr.Error(), Err: runErr,
		}
		_ = a.hooks.OnToolEnd(ctx, endEvent)

		result := errorResult(tc.ID, runErr)
		blocks <- &StreamBlock{
			Type:      BlockToolResult,
			Iteration: iter,
			ToolName:  tc.Name,
			ToolID:    tc.ID,
			Payload:   result.Content,
			IsError:   true,
		}
		return result
	}

	output, runErr := t.Run(ctx, startEvent.Input)

	// ── OnToolEnd ────────────────────────────────────────────────────────────
	endEvent := &hook.ToolEndEvent{
		Iteration: iter,
		ToolName:  tc.Name,
		CallID:    tc.ID,
		Output:    output,
		Err:       runErr,
	}
	_ = a.hooks.OnToolEnd(ctx, endEvent)

	// Emit tool result block
	blocks <- &StreamBlock{
		Type:      BlockToolResult,
		Iteration: iter,
		ToolName:  tc.Name,
		ToolID:    tc.ID,
		Payload:   endEvent.Output,
		IsError:   runErr != nil,
	}

	if runErr != nil {
		return errorResult(tc.ID, runErr)
	}

	return provider.ToolResult{
		ToolCallID: tc.ID,
		Content:    endEvent.Output,
		IsError:    false,
	}
}

// emitError sends an error block and records the agent error via hook.
func (a *Agent) emitError(blocks chan<- *StreamBlock, iter int, err error) {
	blocks <- &StreamBlock{
		Type:      BlockError,
		Iteration: iter,
		Full:      err.Error(),
	}
	_ = a.hooks.OnAgentError(context.Background(), &hook.AgentErrorEvent{Err: err, Iterations: iter})
}
