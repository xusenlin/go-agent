// Package mcp provides an MCP (Model Context Protocol) client that wraps
// remote tools as local tool.Tool implementations. Supports both Stdio
// (local process) and SSE (remote HTTP) transports, auto-detected from
// the server spec.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xusenlin/go-agent/provider"
)

// ─── JSON-RPC primitives ──────────────────────────────────────────────────────

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ─── Transport interface ──────────────────────────────────────────────────────

type transport interface {
	Send(ctx context.Context, req rpcRequest) (*rpcResponse, error)
	Close() error
}

// ─── Stdio transport ──────────────────────────────────────────────────────────

type stdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mu     sync.Mutex
	nextID atomic.Int64
}

func newStdioTransport(command string, proxy *provider.ProxyConfig) (*stdioTransport, error) {
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return nil, fmt.Errorf("mcp stdio: empty command")
	}
	cmd := exec.Command(parts[0], parts[1:]...)
	// inject proxy env vars into child process
	cmd.Env = append(os.Environ(), proxy.Environ()...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp stdio: start process: %w", err)
	}
	return &stdioTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewScanner(stdoutPipe),
	}, nil
}

func (t *stdioTransport) Send(_ context.Context, req rpcRequest) (*rpcResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	if _, err := fmt.Fprintf(t.stdin, "%s\n", data); err != nil {
		return nil, err
	}
	if !t.stdout.Scan() {
		return nil, fmt.Errorf("mcp stdio: connection closed")
	}
	var resp rpcResponse
	if err := json.Unmarshal(t.stdout.Bytes(), &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (t *stdioTransport) Close() error {
	t.stdin.Close()
	return t.cmd.Wait()
}

// ─── SSE transport ────────────────────────────────────────────────────────────

type sseTransport struct {
	baseURL    string
	httpClient *http.Client
	nextID     atomic.Int64
	// SSE endpoint sends events; we derive the POST endpoint from the session
	sessionURL string
	mu         sync.Mutex
}

func newSSETransport(baseURL string, proxy *provider.ProxyConfig) (*sseTransport, error) {
	t := &sseTransport{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: proxy.HTTPClient(),
	}
	// Establish SSE session
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := t.httpClient.Get(t.baseURL + "/sse")
	if err != nil {
		return nil, fmt.Errorf("mcp sse: connect: %w", err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			var event struct{ Endpoint string `json:"endpoint"` }
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err == nil {
				t.sessionURL = t.baseURL + event.Endpoint
				break
			}
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("mcp sse: timeout waiting for session endpoint")
		default:
		}
	}
	if t.sessionURL == "" {
		t.sessionURL = t.baseURL + "/message"
	}
	return t, nil
}

func (t *sseTransport) Send(ctx context.Context, req rpcRequest) (*rpcResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, t.sessionURL, bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := t.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("mcp sse send: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var rpcResp rpcResponse
	if err := json.Unmarshal(raw, &rpcResp); err != nil {
		return nil, err
	}
	return &rpcResp, nil
}

func (t *sseTransport) Close() error { return nil }

// ─── Client ───────────────────────────────────────────────────────────────────

// Client is an MCP client connected to one server.
type Client struct {
	transport transport
}

// NewClient creates a Client, auto-detecting the transport:
//   - If spec starts with "http://" or "https://" → SSE
//   - Otherwise → Stdio (treated as a shell command)
func NewClient(spec string, proxy *provider.ProxyConfig) (*Client, error) {
	var tr transport
	var err error
	if strings.HasPrefix(spec, "http://") || strings.HasPrefix(spec, "https://") {
		tr, err = newSSETransport(spec, proxy)
	} else {
		tr, err = newStdioTransport(spec, proxy)
	}
	if err != nil {
		return nil, err
	}
	c := &Client{transport: tr}
	if err := c.initialize(context.Background()); err != nil {
		_ = tr.Close()
		return nil, err
	}
	return c, nil
}

func (c *Client) initialize(ctx context.Context) error {
	req := rpcRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]any{
			"protocolVersion": "2024-11-05",
			"clientInfo":      map[string]string{"name": "go-agent", "version": "0.1.0"},
			"capabilities":    map[string]any{},
		},
	}
	resp, err := c.transport.Send(ctx, req)
	if err != nil {
		return fmt.Errorf("mcp initialize: %w", err)
	}
	if resp.Error != nil {
		return fmt.Errorf("mcp initialize error: %s", resp.Error.Message)
	}
	// send initialized notification
	notif := rpcRequest{JSONRPC: "2.0", Method: "notifications/initialized"}
	_, _ = c.transport.Send(ctx, notif)
	return nil
}

// ToolInfo describes one tool exposed by the MCP server.
type ToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// Discover fetches the tool list from the MCP server.
func (c *Client) Discover(ctx context.Context) ([]ToolInfo, error) {
	resp, err := c.transport.Send(ctx, rpcRequest{
		JSONRPC: "2.0", ID: 2, Method: "tools/list",
	})
	if err != nil {
		return nil, fmt.Errorf("mcp discover: %w", err)
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("mcp discover error: %s", resp.Error.Message)
	}
	var result struct {
		Tools []ToolInfo `json:"tools"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}
	return result.Tools, nil
}

// Call invokes a tool on the MCP server and returns its text content.
func (c *Client) Call(ctx context.Context, name string, args json.RawMessage) (string, error) {
	var params map[string]any
	_ = json.Unmarshal(args, &params)

	resp, err := c.transport.Send(ctx, rpcRequest{
		JSONRPC: "2.0", ID: 3,
		Method: "tools/call",
		Params: map[string]any{"name": name, "arguments": params},
	})
	if err != nil {
		return "", err
	}
	if resp.Error != nil {
		return "", fmt.Errorf("mcp call %s: %s", name, resp.Error.Message)
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return string(resp.Result), nil
	}
	if result.IsError && len(result.Content) > 0 {
		return "", fmt.Errorf("mcp tool error: %s", result.Content[0].Text)
	}
	var parts []string
	for _, c := range result.Content {
		if c.Type == "text" {
			parts = append(parts, c.Text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// Close shuts down the underlying transport.
func (c *Client) Close() error { return c.transport.Close() }
