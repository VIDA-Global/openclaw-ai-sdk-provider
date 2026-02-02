# @vida-global/openclaw-ai-sdk-provider

Community provider for the [Vercel AI SDK](https://ai-sdk.dev) that targets OpenClaw’s OpenResponses-compatible `/v1/responses` gateway.

## Install

```bash
npm install @vida-global/openclaw-ai-sdk-provider
```

## Basic usage

```ts
import { generateText } from "ai";
import { createOpenClaw } from "@vida-global/openclaw-ai-sdk-provider";

const openclaw = createOpenClaw({
  baseURL: process.env.OPENCLAW_BASE_URL ?? "http://localhost:18789/v1",
  apiKey: process.env.OPENCLAW_TOKEN,
});

const model = openclaw("openclaw:main");

const result = await generateText({
  model,
  prompt: "Hello from OpenClaw!",
});
```

## Streaming

```ts
import { streamText } from "ai";
import { createOpenClaw } from "@vida-global/openclaw-ai-sdk-provider";

const openclaw = createOpenClaw({ apiKey: process.env.OPENCLAW_TOKEN });
const model = openclaw("openclaw:main");

const { textStream } = await streamText({
  model,
  prompt: "Summarize the last 3 messages.",
});
```

## Provider options

OpenClaw-specific options are passed via `providerOptions.openclaw`:

```ts
{
  providerOptions: {
    openclaw: {
      instructions: "System guidance",
      user: "customer:123",          // stable session routing
      reasoningEffort: "low",        // low | medium | high
      reasoningSummary: "concise",   // auto | concise | detailed
      metadata: { source: "demo" },
      maxToolCalls: 3,
      sessionKey: "agent:main:openai:abc123",
      agentId: "main"
    }
  }
}
```

### Environment variables

The provider reads these by default:

- `OPENCLAW_BASE_URL` (default: `http://localhost:18789/v1`)
- `OPENCLAW_API_KEY` or `OPENCLAW_TOKEN` (bearer token)
- `CLAWDBOT_BASE_URL` / `CLAWDBOT_TOKEN` (legacy aliases)

## Tools (function calling)

OpenClaw’s gateway supports OpenResponses function tools. The provider maps AI SDK tools to:

```json
{ "type": "function", "function": { "name", "description", "parameters" } }
```

When the model calls a tool, OpenClaw returns `function_call` items. To continue the turn, send a follow-up `function_call_output` with the tool result. The AI SDK handles this for you when using the standard tool-calling flow.

## Notes

- OpenClaw uses OpenResponses-style items internally; the provider converts AI SDK prompts to OpenResponses `input` items.
- File and image parts are supported; URLs and base64 payloads are forwarded as `input_file` / `input_image`.
- `sessionKey` and `agentId` are forwarded via headers `x-openclaw-session-key` and `x-openclaw-agent-id`.
