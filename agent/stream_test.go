package agent

import (
	"context"
	"testing"

	"github.com/xusenlin/go-agent/hook"
	"github.com/xusenlin/go-agent/provider"
	"github.com/xusenlin/go-agent/tool"
)

// mockProvider implements provider.Provider for testing.
type mockProvider struct {
	chunks []*provider.Chunk
}

func (m *mockProvider) Name() string             { return "mock" }
func (m *mockProvider) Model() string            { return "mock-model" }
func (m *mockProvider) SetModel(string)          {}
func (m *mockProvider) Chat(ctx context.Context, req *provider.Request) (*provider.Response, error) {
	return nil, nil
}

func (m *mockProvider) Stream(ctx context.Context, req *provider.Request) (<-chan *provider.Chunk, error) {
	ch := make(chan *provider.Chunk, len(m.chunks))
	for _, c := range m.chunks {
		ch <- c
	}
	close(ch)
	return ch, nil
}

func newTestAgent(chunks []*provider.Chunk) *Agent {
	return &Agent{
		provider: &mockProvider{chunks: chunks},
		registry: tool.NewRegistry(),
		hooks:    hook.NewChain(),
		maxIter:  10,
	}
}

func collectBlocks(blocks <-chan *StreamBlock) []*StreamBlock {
	var result []*StreamBlock
	for b := range blocks {
		result = append(result, b)
	}
	return result
}

func TestStream_ThinkThenToolCall(t *testing.T) {
	// 模拟: 思考 -> 工具调用（没有回答）
	chunks := []*provider.Chunk{
		{Delta: "Let me ", IsThinking: true},
		{Delta: "check the weather.", IsThinking: true},
		{Done: true, Delta: `{"StopReason":"tool_use","ToolCalls":[{"ID":"call_1","Name":"get_weather","Input":null}]}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	// 验证: 应该有 ThinkStart, ThinkStream x2, ThinkEnd, 没有 TextStart/TextEnd
	expected := []BlockType{
		BlockThinkStart,
		BlockThinkStream,
		BlockThinkStream,
		BlockThinkEnd,
	}

	if len(blocks) != len(expected) {
		t.Fatalf("expected %d blocks, got %d", len(expected), len(blocks))
	}

	for i, b := range blocks {
		if b.Type != expected[i] {
			t.Errorf("block[%d]: expected %q, got %q", i, expected[i], b.Type)
		}
	}
}

func TestStream_ThinkThenText(t *testing.T) {
	// 模拟: 思考 -> 回答（没有工具调用）
	chunks := []*provider.Chunk{
		{Delta: "Thinking...", IsThinking: true},
		{Delta: "The answer is 42.", IsThinking: false},
		{Done: true, Delta: `{"StopReason":"end_turn"}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	// 验证 block 类型序列
	expected := []BlockType{
		BlockThinkStart,
		BlockThinkStream,
		BlockThinkEnd,
		BlockTextStart,
		BlockTextStream,
		BlockTextEnd,
	}

	if len(blocks) != len(expected) {
		t.Fatalf("expected %d blocks, got %d", len(expected), len(blocks))
	}

	for i, b := range blocks {
		if b.Type != expected[i] {
			t.Errorf("block[%d]: expected %q, got %q", i, expected[i], b.Type)
		}
	}

	// 验证 ThinkEnd 的 Full
	for _, b := range blocks {
		if b.Type == BlockThinkEnd {
			if b.Full != "Thinking..." {
				t.Errorf("ThinkEnd.Full: expected %q, got %q", "Thinking...", b.Full)
			}
		}
		if b.Type == BlockTextEnd {
			if b.Full != "The answer is 42." {
				t.Errorf("TextEnd.Full: expected %q, got %q", "The answer is 42.", b.Full)
			}
		}
	}
}

func TestStream_TextOnly(t *testing.T) {
	// 模拟: 直接回答（没有思考）
	chunks := []*provider.Chunk{
		{Delta: "Hello! "},
		{Delta: "How can I help?"},
		{Done: true, Delta: `{"StopReason":"end_turn"}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	// 验证: 不应该有 ThinkStart
	for _, b := range blocks {
		if b.Type == BlockThinkStart || b.Type == BlockThinkStream || b.Type == BlockThinkEnd {
			t.Errorf("unexpected think block: %s", b.Type)
		}
	}

	// 验证: 应该有 TextStart, TextStream, TextEnd
	expected := []BlockType{
		BlockTextStart,
		BlockTextStream,
		BlockTextStream,
		BlockTextEnd,
	}

	if len(blocks) != len(expected) {
		t.Fatalf("expected %d blocks, got %d", len(expected), len(blocks))
	}

	for i, b := range blocks {
		if b.Type != expected[i] {
			t.Errorf("block[%d]: expected %q, got %q", i, expected[i], b.Type)
		}
	}
}

func TestStream_MultipleThinkBlocks(t *testing.T) {
	// 模拟: 思考 -> 工具 -> 思考 -> 回答
	// 第一次流
	chunks1 := []*provider.Chunk{
		{Delta: "Need to call tool.", IsThinking: true},
		{Done: true, Delta: `{"StopReason":"tool_use","ToolCalls":[{"ID":"call_1","Name":"get_weather","Input":null}]}`},
	}

	a := newTestAgent(chunks1)
	blocksCh1 := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh1)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh1)

	blocks1 := collectBlocks(blocksCh1)

	// 验证第一次流
	expected1 := []BlockType{
		BlockThinkStart,
		BlockThinkStream,
		BlockThinkEnd,
	}

	if len(blocks1) != len(expected1) {
		t.Fatalf("stream1: expected %d blocks, got %d", len(expected1), len(blocks1))
	}

	for i, b := range blocks1 {
		if b.Type != expected1[i] {
			t.Errorf("stream1 block[%d]: expected %q, got %q", i, expected1[i], b.Type)
		}
	}

	// 第二次流
	chunks2 := []*provider.Chunk{
		{Delta: "Got the result.", IsThinking: true},
		{Delta: "The weather is sunny.", IsThinking: false},
		{Done: true, Delta: `{"StopReason":"end_turn"}`},
	}

	a.provider = &mockProvider{chunks: chunks2}
	blocksCh2 := make(chan *StreamBlock, 100)
	_, _, err = a.streamAndAssembleStream(context.Background(), &provider.Request{}, 2, blocksCh2)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh2)

	blocks2 := collectBlocks(blocksCh2)

	// 验证第二次流
	expected2 := []BlockType{
		BlockThinkStart,
		BlockThinkStream,
		BlockThinkEnd,
		BlockTextStart,
		BlockTextStream,
		BlockTextEnd,
	}

	if len(blocks2) != len(expected2) {
		t.Fatalf("stream2: expected %d blocks, got %d", len(expected2), len(blocks2))
	}

	for i, b := range blocks2 {
		if b.Type != expected2[i] {
			t.Errorf("stream2 block[%d]: expected %q, got %q", i, expected2[i], b.Type)
		}
	}
}

func TestStream_ToolCallOnly(t *testing.T) {
	// 模拟: 直接工具调用（没有思考也没有回答）
	chunks := []*provider.Chunk{
		{Done: true, Delta: `{"StopReason":"tool_use","ToolCalls":[{"ID":"call_1","Name":"get_weather","Input":null}]}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	// 验证: 不应该有任何 think 或 text block
	for _, b := range blocks {
		if b.Type == BlockThinkStart || b.Type == BlockThinkStream || b.Type == BlockThinkEnd ||
			b.Type == BlockTextStart || b.Type == BlockTextStream || b.Type == BlockTextEnd {
			t.Errorf("unexpected block: %s", b.Type)
		}
	}
}

func TestStream_EmptyTextIgnored(t *testing.T) {
	// 模拟: 思考后直接工具调用，中间有空白内容
	chunks := []*provider.Chunk{
		{Delta: "Thinking...", IsThinking: true},
		{Delta: " ", IsThinking: false},  // 空白内容
		{Delta: "\n", IsThinking: false}, // 换行
		{Done: true, Delta: `{"StopReason":"tool_use","ToolCalls":[{"ID":"call_1","Name":"get_weather","Input":null}]}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	// 验证: 不应该有 TextStart/TextEnd（因为空白内容被忽略）
	for _, b := range blocks {
		if b.Type == BlockTextStart || b.Type == BlockTextStream || b.Type == BlockTextEnd {
			t.Errorf("unexpected text block: %s", b.Type)
		}
	}
}

func TestStream_EndContent(t *testing.T) {
	// 模拟: 思考 -> 回答，验证 Full
	chunks := []*provider.Chunk{
		{Delta: "Part1 ", IsThinking: true},
		{Delta: "Part2", IsThinking: true},
		{Delta: "Answer1 ", IsThinking: false},
		{Delta: "Answer2", IsThinking: false},
		{Done: true, Delta: `{"StopReason":"end_turn"}`},
	}

	a := newTestAgent(chunks)
	blocksCh := make(chan *StreamBlock, 100)
	_, _, err := a.streamAndAssembleStream(context.Background(), &provider.Request{}, 1, blocksCh)
	if err != nil {
		t.Fatal(err)
	}
	close(blocksCh)

	blocks := collectBlocks(blocksCh)

	for _, b := range blocks {
		switch b.Type {
		case BlockThinkEnd:
			if b.Full != "Part1 Part2" {
				t.Errorf("ThinkEnd.Full: expected %q, got %q", "Part1 Part2", b.Full)
			}
		case BlockTextEnd:
			if b.Full != "Answer1 Answer2" {
				t.Errorf("TextEnd.Full: expected %q, got %q", "Answer1 Answer2", b.Full)
			}
		}
	}
}
