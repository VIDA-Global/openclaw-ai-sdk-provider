# openclaw-ai-sdk-provider

A custom Vercel AI SDK provider for OpenClaw's `/v1/responses` API.

## Usage

```ts
import { createOpenClaw } from "openclaw-ai-sdk-provider";

const openclaw = createOpenClaw({
  baseURL: process.env.OPENCLAW_BASE_URL ?? "http://localhost:18789/v1",
  apiKey: process.env.OPENCLAW_TOKEN,
});

const model = openclaw("openclaw:main");
```

### Provider options

The provider supports OpenClaw-specific options via `providerOptions.openclaw`:

```ts
{
  providerOptions: {
    openclaw: {
      instructions: "system guidance",
      user: "vida:user:123",
      reasoningEffort: "low",
      reasoningSummary: "concise",
      metadata: { source: "vida" },
      maxToolCalls: 3,
      sessionKey: "agent:main:openai:abc123",
      agentId: "main"
    }
  }
}
```

## Notes

- This provider speaks OpenClaw's OpenResponses schema (input items must include `type: "message"`).
- OpenClaw currently does not stream tool logs over `/v1/responses`; tool calls are only returned in the `response.output` items when OpenClaw stops for a tool call.
