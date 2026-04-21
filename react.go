package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/xusenlin/go-agent/hook"
	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/tool"
	plantool "github.com/xusenlin/go-agent/tool/plan"
)

// Result is the output of a completed ReAct run.
type Result struct {
	Output     string
	Iterations int
}

// Run starts the ReAct loop for the given user input.
//
// The loop:
//  1. Calls the LLM (streaming internally; hooks see the full assembled response)
//  2. If stop_reason == "end_turn" → done, return the assistant's message
//  3. For each tool_call in the response:
//     a. Fire OnToolStart (hooks may modify input)
//     b. Execute the tool
//     c. Fire OnToolEnd (hooks may modify output)
//     d. If tool == create_plan → also fire OnPlanCreated
//  4. Append assistant message + tool results to history, loop back to 1
//  5. Abort with error if maxIter is exceeded
func (a *Agent) Run(ctx context.Context, input string) (*Result, error) {
	// ── Setup ────────────────────────────────────────────────────────────────
	toolDefs := tool.ToProviderDefs(a.registry.All())

	messages := []provider.Message{
		{Role: provider.RoleUser, Content: input},
	}

	// ── OnAgentStart ─────────────────────────────────────────────────────────
	if err := a.hooks.OnAgentStart(ctx, &hook.AgentStartEvent{Input: input}); err != nil {
		return nil, fmt.Errorf("hook OnAgentStart: %w", err)
	}

	// ── ReAct loop ────────────────────────────────────────────────────────────
	for iter := 1; iter <= a.maxIter; iter++ {
		req := &provider.Request{
			System:   a.systemPrompt,
			Messages: messages,
			Tools:    toolDefs,
			Stream:   true,
		}

		// ── Think ────────────────────────────────────────────────────────────
		if err := a.hooks.OnThinkStart(ctx, &hook.ThinkStartEvent{
			Iteration: iter,
			Messages:  messages,
		}); err != nil {
			return nil, fmt.Errorf("hook OnThinkStart: %w", err)
		}

		resp, streamText, err := a.streamAndAssemble(ctx, req)
		if err != nil {
			_ = a.hooks.OnAgentError(ctx, &hook.AgentErrorEvent{Err: err, Iterations: iter})
			return nil, fmt.Errorf("agent think iter %d: %w", iter, err)
		}

		if err := a.hooks.OnThinkEnd(ctx, &hook.ThinkEndEvent{
			Iteration:    iter,
			Response:     resp,
			StreamTokens: streamText,
		}); err != nil {
			return nil, fmt.Errorf("hook OnThinkEnd: %w", err)
		}

		// ── End turn: no more tool calls ─────────────────────────────────────
		if resp.StopReason == "end_turn" || len(resp.ToolCalls) == 0 {
			output := resp.Content
			_ = a.hooks.OnAgentFinish(ctx, &hook.AgentFinishEvent{
				Output:     output,
				Iterations: iter,
			})
			return &Result{Output: output, Iterations: iter}, nil
		}

		// ── Append assistant message (with tool calls) to history ─────────────
		messages = append(messages, provider.Message{
			Role:      provider.RoleAssistant,
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		// ── Execute tool calls ────────────────────────────────────────────────
		var toolResults []provider.ToolResult

		for _, tc := range resp.ToolCalls {
			toolResult, err := a.executeTool(ctx, iter, tc)
			toolResults = append(toolResults, toolResult)
			// errors from tools are formatted as tool results (not fatal),
			// so the LLM can decide how to handle them.
			_ = err
		}

		// ── Append tool results as a single user message ──────────────────────
		messages = append(messages, provider.Message{
			Role:        provider.RoleUser,
			ToolResults: toolResults,
		})
	}

	// ── Max iterations exceeded ───────────────────────────────────────────────
	err := fmt.Errorf("agent: max iterations (%d) exceeded", a.maxIter)
	_ = a.hooks.OnAgentError(ctx, &hook.AgentErrorEvent{Err: err, Iterations: a.maxIter})
	return nil, err
}

// ─── streaming + assemble ─────────────────────────────────────────────────────

// streamAndAssemble calls the provider in streaming mode, forwards deltas
// to the stream channel (if set), and returns the fully assembled Response.
func (a *Agent) streamAndAssemble(ctx context.Context, req *provider.Request) (*provider.Response, string, error) {
	streamCh, err := a.provider.Stream(ctx, req)
	if err != nil {
		return nil, "", err
	}

	var textBuilder strings.Builder

	for chunk := range streamCh {
		if !chunk.Done {
			textBuilder.WriteString(chunk.Delta)
			continue
		}
		// Last chunk carries the full assembled response as JSON in Delta.
		if chunk.Delta != "" {
			var resp provider.Response
			if jsonErr := json.Unmarshal([]byte(chunk.Delta), &resp); jsonErr == nil {
				return &resp, textBuilder.String(), nil
			}
		}
	}

	// Fallback: if provider doesn't encode the final response in the last chunk,
	// use Chat() to get a clean response.
	req.Stream = false
	resp, err := a.provider.Chat(ctx, req)
	return resp, textBuilder.String(), err
}

// ─── tool execution ───────────────────────────────────────────────────────────

func (a *Agent) executeTool(ctx context.Context, iter int, tc provider.ToolCall) (provider.ToolResult, error) {
	// ── OnToolStart (hooks may modify input) ──────────────────────────────────
	startEvent := &hook.ToolStartEvent{
		Iteration: iter,
		ToolName:  tc.Name,
		CallID:    tc.ID,
		Input:     tc.Input,
	}
	if err := a.hooks.OnToolStart(ctx, startEvent); err != nil {
		return errorResult(tc.ID, fmt.Errorf("hook OnToolStart: %w", err)), err
	}
	// Use potentially-modified input from hook
	effectiveInput := startEvent.Input

	// ── Look up and run tool ──────────────────────────────────────────────────
	t, ok := a.registry.Get(tc.Name)
	if !ok {
		runErr := fmt.Errorf("unknown tool: %q", tc.Name)
		endEvent := &hook.ToolEndEvent{
			Iteration: iter, ToolName: tc.Name, CallID: tc.ID,
			Output: runErr.Error(), Err: runErr,
		}
		_ = a.hooks.OnToolEnd(ctx, endEvent)
		return errorResult(tc.ID, runErr), runErr
	}

	output, runErr := t.Run(ctx, effectiveInput)

	// ── OnToolEnd (hooks may modify output) ───────────────────────────────────
	endEvent := &hook.ToolEndEvent{
		Iteration: iter,
		ToolName:  tc.Name,
		CallID:    tc.ID,
		Output:    output,
		Err:       runErr,
	}
	_ = a.hooks.OnToolEnd(ctx, endEvent)
	// Use potentially-modified output from hook
	effectiveOutput := endEvent.Output

	if runErr != nil {
		return errorResult(tc.ID, runErr), runErr
	}

	// ── OnPlanCreated (plan tool special case) ────────────────────────────────
	if tc.Name == plantool.ToolName {
		if steps, ok := plantool.ParseResult(effectiveOutput); ok {
			planEvent := &hook.PlanCreatedEvent{Iteration: iter, Steps: steps}
			_ = a.hooks.OnPlanCreated(ctx, planEvent)
		}
	}

	return provider.ToolResult{
		ToolCallID: tc.ID,
		Content:    effectiveOutput,
		IsError:    false,
	}, nil
}

// ─── helpers ──────────────────────────────────────────────────────────────────

func errorResult(callID string, err error) provider.ToolResult {
	return provider.ToolResult{
		ToolCallID: callID,
		Content:    "Error: " + err.Error(),
		IsError:    true,
	}
}

// buildCtx returns a background context used during Builder.Build() for MCP discovery.
// It is a method on Builder (defined here to keep react.go focused on the loop).
func (b *Builder) buildCtx() context.Context {
	return context.Background()
}
