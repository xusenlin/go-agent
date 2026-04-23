# Streaming API

`RunStream` returns a channel of `StreamBlock` for real-time streaming.

## StreamBlock Fields

Each `StreamBlock` has three mutually exclusive content fields, determined by `Type`:

| Field | Type | Description | Block Types |
|-------|------|-------------|-------------|
| `Delta` | string | Streaming increment (real-time rendering) | ThinkStream, TextStream |
| `Full` | string | Complete content (for storage/persistence) | ThinkEnd, TextEnd, Error |
| `Payload` | string | Event data (JSON or output) | ToolCall, ToolResult |

## Block Types

### Streaming Blocks (use `Delta`)

Real-time incremental content for UI rendering:

```go
case agent.BlockThinkStream:
    ui.AppendThinking(block.Delta)  // incremental thinking text

case agent.BlockTextStream:
    ui.AppendText(block.Delta)      // incremental response text
```

### End Blocks (use `Full`)

Complete content when a block finishes:

```go
case agent.BlockThinkEnd:
    db.SaveThinking(messageID, block.Full)  // complete thinking text

case agent.BlockTextEnd:
    db.SaveText(messageID, block.Full)      // complete response text

case agent.BlockError:
    db.SaveError(messageID, block.Full)     // error message
```

### Event Blocks (use `Payload`)

Tool invocation and results:

```go
case agent.BlockToolCall:
    // block.Payload contains input JSON
    db.SaveToolCall(messageID, block.ToolName, block.ToolID, block.Payload)

case agent.BlockToolResult:
    // block.Payload contains output text
    db.SaveToolResult(messageID, block.ToolName, block.ToolID, block.Payload, block.IsError)
```

### Token Block (no content field)

Final token usage statistics:

```go
case agent.BlockFinish:
    db.UpdateTokens(messageID, block.InputTokens, block.OutputTokens)
```

## Complete Example

```go
streamCh, err := agent.RunStream(ctx, userInput)
if err != nil {
    log.Fatal(err)
}

for block := range streamCh {
    switch block.Type {
    // Streaming - use Delta
    case agent.BlockThinkStream:
        ui.AppendThinking(block.Delta)
    case agent.BlockTextStream:
        ui.AppendText(block.Delta)

    // Complete content - use Full
    case agent.BlockThinkEnd:
        db.SaveThinking(messageID, block.Full)
    case agent.BlockTextEnd:
        db.SaveText(messageID, block.Full)
    case agent.BlockError:
        db.SaveError(messageID, block.Full)

    // Events - use Payload
    case agent.BlockToolCall:
        db.SaveToolCall(messageID, block.ToolName, block.ToolID, block.Payload)
    case agent.BlockToolResult:
        db.SaveToolResult(messageID, block.ToolName, block.ToolID, block.Payload, block.IsError)

    // Token stats
    case agent.BlockFinish:
        db.UpdateTokens(messageID, block.InputTokens, block.OutputTokens)
    }
}
```

## Block Type Sequence

A typical streaming session produces blocks in this order:

```
ThinkStart
ThinkStream (multiple)
ThinkEnd
TextStart
TextStream (multiple)
TextEnd
Finish
```

With tool calls:

```
ThinkStart
ThinkStream (multiple)
ThinkEnd
ToolCall
ToolResult
ThinkStart
ThinkStream (multiple)
ThinkEnd
TextStart
TextStream (multiple)
TextEnd
Finish
```
