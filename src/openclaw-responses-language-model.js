import { APICallError } from "@ai-sdk/provider";
import {
  combineHeaders,
  convertToBase64,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod";

const openclawProviderOptionsSchema = z
  .object({
    instructions: z.string().optional(),
    user: z.string().optional(),
    reasoningEffort: z.enum(["low", "medium", "high"]).optional(),
    reasoningSummary: z.enum(["auto", "concise", "detailed"]).optional(),
    metadata: z.record(z.string(), z.string()).optional(),
    maxToolCalls: z.number().int().positive().optional(),
    sessionKey: z.string().optional(),
    agentId: z.string().optional(),
  })
  .optional();

const openclawErrorSchema = z.object({
  error: z.object({
    message: z.string(),
    type: z.string().optional(),
  }),
});

const openclawUsageSchema = z.object({
  input_tokens: z.number().int().nonnegative(),
  output_tokens: z.number().int().nonnegative(),
  total_tokens: z.number().int().nonnegative(),
});

const openclawOutputTextSchema = z.object({
  type: z.literal("output_text"),
  text: z.string(),
});

const openclawOutputItemSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("message"),
    id: z.string(),
    role: z.literal("assistant"),
    content: z.array(openclawOutputTextSchema),
    status: z.enum(["in_progress", "completed"]).optional(),
  }),
  z.object({
    type: z.literal("function_call"),
    id: z.string(),
    call_id: z.string(),
    name: z.string(),
    arguments: z.string(),
    status: z.enum(["in_progress", "completed"]).optional(),
  }),
  z.object({
    type: z.literal("reasoning"),
    id: z.string(),
    content: z.string().optional(),
    summary: z.string().optional(),
  }),
]);

const openclawResponseSchema = z.object({
  id: z.string(),
  object: z.literal("response"),
  created_at: z.number().int(),
  status: z.enum(["in_progress", "completed", "failed", "cancelled", "incomplete"]),
  model: z.string(),
  output: z.array(openclawOutputItemSchema),
  usage: openclawUsageSchema,
  error: z
    .object({
      code: z.string(),
      message: z.string(),
    })
    .optional(),
});

const openclawStreamEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("response.created"),
    response: openclawResponseSchema,
  }),
  z.object({
    type: z.literal("response.in_progress"),
    response: openclawResponseSchema,
  }),
  z.object({
    type: z.literal("response.completed"),
    response: openclawResponseSchema,
  }),
  z.object({
    type: z.literal("response.failed"),
    response: openclawResponseSchema,
  }),
  z.object({
    type: z.literal("response.output_text.delta"),
    item_id: z.string(),
    output_index: z.number().int(),
    content_index: z.number().int(),
    delta: z.string(),
  }),
  z.object({
    type: z.literal("response.output_text.done"),
    item_id: z.string(),
    output_index: z.number().int(),
    content_index: z.number().int(),
    text: z.string(),
  }),
  z.object({
    type: z.literal("response.output_item.added"),
    output_index: z.number().int(),
    item: openclawOutputItemSchema,
  }),
  z.object({
    type: z.literal("response.output_item.done"),
    output_index: z.number().int(),
    item: openclawOutputItemSchema,
  }),
  z.object({
    type: z.literal("response.content_part.added"),
    item_id: z.string(),
    output_index: z.number().int(),
    content_index: z.number().int(),
    part: openclawOutputTextSchema,
  }),
  z.object({
    type: z.literal("response.content_part.done"),
    item_id: z.string(),
    output_index: z.number().int(),
    content_index: z.number().int(),
    part: openclawOutputTextSchema,
  }),
]);

const openclawFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: openclawErrorSchema,
  errorToMessage: (data) => data.error.message,
});

function mapUsage(usage) {
  return {
    inputTokens: {
      total: usage.input_tokens,
      noCache: usage.input_tokens,
      cacheRead: 0,
      cacheWrite: 0,
    },
    outputTokens: {
      total: usage.output_tokens,
      text: usage.output_tokens,
      reasoning: 0,
    },
    raw: usage,
  };
}

function mapFinishReason({ status, sawToolCall }) {
  if (sawToolCall || status === "incomplete") {
    return { unified: "tool-calls", raw: "tool_calls" };
  }
  switch (status) {
    case "completed":
      return { unified: "stop", raw: "completed" };
    case "failed":
      return { unified: "error", raw: "failed" };
    default:
      return { unified: "other", raw: status };
  }
}

function toToolOutputString(output) {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value;
    case "json":
    case "error-json":
      return JSON.stringify(output.value);
    case "execution-denied":
      return output.reason ?? "Tool execution denied.";
    case "content":
      return JSON.stringify(output.value);
    default:
      return JSON.stringify(output);
  }
}

function isHttpUrl(value) {
  return /^https?:\/\//i.test(value);
}

function normalizeContentParts(message) {
  const content = message?.content;
  if (Array.isArray(content)) return content;
  if (typeof content === "string") {
    return [{ type: "text", text: content }];
  }
  return [];
}

function convertPromptToInput(prompt) {
  const input = [];

  for (const message of prompt) {
    if (message.role === "system" || message.role === "developer") {
      const text = normalizeContentParts(message)
        .filter((part) => part && part.type === "text")
        .map((part) => part.text)
        .join("\n");
      if (text) {
        input.push({ type: "message", role: message.role, content: text });
      }
      continue;
    }

    if (message.role === "user" || message.role === "assistant") {
      const contentParts = [];
      const assistant = message.role === "assistant";

      for (const part of normalizeContentParts(message)) {
        if (part.type === "text") {
          contentParts.push({
            type: assistant ? "output_text" : "input_text",
            text: part.text,
          });
          continue;
        }
        if (part.type === "file") {
          const mediaType = part.mediaType === "image/*" ? "image/jpeg" : part.mediaType;
          const isImage = mediaType.startsWith("image/");
          if (part.data instanceof URL) {
            contentParts.push(
              isImage
                ? { type: "input_image", source: { type: "url", url: part.data.toString() } }
                : { type: "input_file", source: { type: "url", url: part.data.toString() } },
            );
            continue;
          }
          if (typeof part.data === "string" && isHttpUrl(part.data)) {
            contentParts.push(
              isImage
                ? { type: "input_image", source: { type: "url", url: part.data } }
                : { type: "input_file", source: { type: "url", url: part.data } },
            );
            continue;
          }
          const data = typeof part.data === "string" ? part.data : convertToBase64(part.data);
          contentParts.push(
            isImage
              ? {
                  type: "input_image",
                  source: { type: "base64", media_type: mediaType, data },
                }
              : {
                  type: "input_file",
                  source: {
                    type: "base64",
                    media_type: mediaType,
                    data,
                    filename: part.filename,
                  },
                },
          );
          continue;
        }
        if (part.type === "tool-call") {
          input.push({
            type: "function_call",
            call_id: part.toolCallId,
            name: part.toolName,
            arguments: part.input,
          });
          continue;
        }
        if (part.type === "reasoning") {
          input.push({
            type: "reasoning",
            content: part.text,
          });
        }
      }

      if (contentParts.length > 0) {
        input.push({
          type: "message",
          role: message.role,
          content: contentParts,
        });
      }
      continue;
    }

    if (message.role === "tool") {
      const toolParts = Array.isArray(message.content) ? message.content : [];
      for (const part of toolParts) {
        if (part.type === "tool-approval-response") {
          continue;
        }
        input.push({
          type: "function_call_output",
          call_id: part.toolCallId,
          output: toToolOutputString(part.output),
        });
      }
    }
  }

  return input;
}

function mapTools(tools, warnings) {
  if (!tools || tools.length === 0) return undefined;
  const functionTools = tools.filter((tool) => tool.type === "function");
  const providerTools = tools.filter((tool) => tool.type === "provider");

  if (providerTools.length > 0) {
    warnings.push({
      type: "unsupported",
      feature: "providerTools",
      details: "OpenClaw responses only support function tools",
    });
  }

  if (functionTools.length === 0) return undefined;

  return functionTools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
}

function mapToolChoice(toolChoice) {
  if (!toolChoice) return undefined;
  switch (toolChoice.type) {
    case "auto":
      return "auto";
    case "none":
      return "none";
    case "required":
      return "required";
    case "tool":
      return {
        type: "function",
        function: { name: toolChoice.toolName },
      };
    default:
      return undefined;
  }
}

function buildHeaders(baseHeaders, headers, providerOptions) {
  const extra = {};
  if (providerOptions?.sessionKey) {
    extra["x-openclaw-session-key"] = providerOptions.sessionKey;
  }
  if (providerOptions?.agentId) {
    extra["x-openclaw-agent-id"] = providerOptions.agentId;
  }
  return combineHeaders(baseHeaders, headers, extra);
}

export class OpenClawResponsesLanguageModel {
  specificationVersion = "v3";
  supportedUrls = {
    "image/*": [/^https?:\/\/.*$/],
    "application/pdf": [/^https?:\/\/.*$/],
  };

  constructor(modelId, config) {
    this.modelId = modelId;
    this.config = config;
  }

  get provider() {
    return this.config.providerName;
  }

  async getArgs(options) {
    const warnings = [];

    if (options.topK != null) {
      warnings.push({ type: "unsupported", feature: "topK" });
    }
    if (options.stopSequences != null) {
      warnings.push({ type: "unsupported", feature: "stopSequences" });
    }
    if (options.presencePenalty != null) {
      warnings.push({ type: "unsupported", feature: "presencePenalty" });
    }
    if (options.frequencyPenalty != null) {
      warnings.push({ type: "unsupported", feature: "frequencyPenalty" });
    }
    if (options.seed != null) {
      warnings.push({ type: "unsupported", feature: "seed" });
    }
    if (options.responseFormat != null) {
      warnings.push({ type: "unsupported", feature: "responseFormat" });
    }

    const providerOptions = await parseProviderOptions({
      provider: "openclaw",
      providerOptions: options.providerOptions,
      schema: openclawProviderOptionsSchema,
    });

    const input = convertPromptToInput(options.prompt);
    const tools = mapTools(options.tools, warnings);
    const toolChoice = mapToolChoice(options.toolChoice);

    const body = {
      model: this.modelId,
      input,
      temperature: options.temperature,
      top_p: options.topP,
      max_output_tokens: options.maxOutputTokens,
      tools,
      tool_choice: toolChoice,
      instructions: providerOptions?.instructions,
      user: providerOptions?.user,
      metadata: providerOptions?.metadata,
      max_tool_calls: providerOptions?.maxToolCalls,
      reasoning:
        providerOptions?.reasoningEffort || providerOptions?.reasoningSummary
          ? {
              effort: providerOptions?.reasoningEffort,
              summary: providerOptions?.reasoningSummary,
            }
          : undefined,
    };

    return { body, warnings, providerOptions };
  }

  async doGenerate(options) {
    const { body, warnings, providerOptions } = await this.getArgs(options);
    const url = `${this.config.baseURL.replace(/\/$/, "")}/responses`;

    const { responseHeaders, value: response, rawValue } = await postJsonToApi({
      url,
      headers: buildHeaders(this.config.headers(), options.headers, providerOptions),
      body,
      failedResponseHandler: openclawFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(openclawResponseSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    if (response.error) {
      throw new APICallError({
        message: response.error.message,
        url,
        requestBodyValues: body,
        statusCode: 400,
        responseHeaders,
        responseBody: rawValue,
        isRetryable: false,
      });
    }

    const content = [];
    let sawToolCall = false;

    for (const item of response.output) {
      if (item.type === "message") {
        for (const part of item.content) {
          content.push({
            type: "text",
            text: part.text,
            providerMetadata: {
              [this.config.providerName]: { itemId: item.id },
            },
          });
        }
      } else if (item.type === "function_call") {
        sawToolCall = true;
        content.push({
          type: "tool-call",
          toolCallId: item.call_id,
          toolName: item.name,
          input: item.arguments,
          providerMetadata: {
            [this.config.providerName]: { itemId: item.id },
          },
        });
      } else if (item.type === "reasoning") {
        const text = item.summary ?? item.content;
        if (text) {
          content.push({
            type: "reasoning",
            text,
            providerMetadata: {
              [this.config.providerName]: { itemId: item.id },
            },
          });
        }
      }
    }

    const finishReason = mapFinishReason({ status: response.status, sawToolCall });

    return {
      content,
      finishReason,
      usage: mapUsage(response.usage),
      request: { body },
      response: {
        id: response.id,
        timestamp: new Date(response.created_at * 1e3),
        modelId: response.model,
        headers: responseHeaders,
        body: rawValue,
      },
      providerMetadata: {
        [this.config.providerName]: { responseId: response.id },
      },
      warnings,
    };
  }

  async doStream(options) {
    const { body, warnings, providerOptions } = await this.getArgs(options);
    const url = `${this.config.baseURL.replace(/\/$/, "")}/responses`;
    const providerName = this.config.providerName;

    const { responseHeaders, value: response } = await postJsonToApi({
      url,
      headers: buildHeaders(this.config.headers(), options.headers, providerOptions),
      body: { ...body, stream: true },
      failedResponseHandler: openclawFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(openclawStreamEventSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let responseId;
    let sawToolCall = false;
    let usage;
    let finishReason;
    let activeTextId = null;

    return {
      stream: response.pipeThrough(
        new TransformStream({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings });
          },
          transform(chunk, controller) {
            if (options.includeRawChunks) {
              controller.enqueue({ type: "raw", rawValue: chunk.rawValue });
            }

            if (!chunk.success) {
              finishReason = { unified: "error", raw: "error" };
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const event = chunk.value;

            switch (event.type) {
              case "response.created": {
                responseId = event.response.id;
                controller.enqueue({
                  type: "response-metadata",
                  id: event.response.id,
                  timestamp: new Date(event.response.created_at * 1e3),
                  modelId: event.response.model,
                  headers: responseHeaders,
                });
                return;
              }
              case "response.output_text.delta": {
                if (!activeTextId) {
                  activeTextId = event.item_id;
                  controller.enqueue({ type: "text-start", id: activeTextId });
                }
                controller.enqueue({
                  type: "text-delta",
                  id: activeTextId,
                  delta: event.delta,
                });
                return;
              }
              case "response.output_item.added":
              case "response.output_item.done": {
                const item = event.item;
                if (item.type === "function_call") {
                  sawToolCall = true;
                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: item.call_id,
                    toolName: item.name,
                    input: item.arguments,
                    providerMetadata: {
                      [providerName]: { itemId: item.id },
                    },
                  });
                }
                if (item.type === "reasoning") {
                  const text = item.summary ?? item.content;
                  if (text) {
                    controller.enqueue({ type: "reasoning-start", id: item.id });
                    controller.enqueue({ type: "reasoning-delta", id: item.id, delta: text });
                    controller.enqueue({ type: "reasoning-end", id: item.id });
                  }
                }
                return;
              }
              case "response.completed":
              case "response.failed": {
                usage = mapUsage(event.response.usage);
                finishReason = mapFinishReason({
                  status: event.response.status,
                  sawToolCall,
                });
                return;
              }
              default:
                return;
            }
          },
          flush(controller) {
            if (activeTextId) {
              controller.enqueue({ type: "text-end", id: activeTextId });
            }
            if (!usage) {
              return;
            }
            controller.enqueue({
              type: "finish",
              usage,
              finishReason: finishReason ?? { unified: "other", raw: "unknown" },
              providerMetadata: responseId
                ? { [providerName]: { responseId } }
                : undefined,
            });
          },
        }),
      ),
    };
  }
}
