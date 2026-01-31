import {
  APICallError,
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
  type LanguageModelV3FinishReason,
  type LanguageModelV3Prompt,
  type LanguageModelV3ToolResultPart,
  type LanguageModelV3Usage,
  type SharedV3Headers,
  type SharedV3ProviderMetadata,
  type SharedV3Warning,
} from "@ai-sdk/provider";
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

type OpenClawResponsesConfig = {
  baseURL: string;
  headers: () => Record<string, string>;
  fetch?: typeof fetch;
  generateId?: () => string;
  providerName: string;
};

type OpenClawProviderOptions = {
  instructions?: string;
  user?: string;
  reasoningEffort?: "low" | "medium" | "high";
  reasoningSummary?: "auto" | "concise" | "detailed";
  metadata?: Record<string, string>;
  maxToolCalls?: number;
  sessionKey?: string;
  agentId?: string;
};

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

type OpenClawInputPart =
  | { type: "input_text"; text: string }
  | {
      type: "input_image";
      source:
        | { type: "url"; url: string }
        | { type: "base64"; media_type: string; data: string };
    }
  | {
      type: "input_file";
      source:
        | { type: "url"; url: string }
        | { type: "base64"; media_type: string; data: string; filename?: string };
    }
  | { type: "output_text"; text: string };

type OpenClawInputItem =
  | {
      type: "message";
      role: "system" | "developer" | "user" | "assistant";
      content: string | OpenClawInputPart[];
    }
  | {
      type: "function_call";
      id?: string;
      call_id?: string;
      name: string;
      arguments: string;
    }
  | {
      type: "function_call_output";
      call_id: string;
      output: string;
    }
  | {
      type: "reasoning";
      content?: string;
      summary?: string;
    }
  | {
      type: "item_reference";
      id: string;
    };

function mapUsage(usage: z.infer<typeof openclawUsageSchema>): LanguageModelV3Usage {
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

function mapFinishReason(params: {
  status: z.infer<typeof openclawResponseSchema>["status"];
  sawToolCall: boolean;
}): LanguageModelV3FinishReason {
  if (params.sawToolCall || params.status === "incomplete") {
    return { unified: "tool-calls", raw: "tool_calls" };
  }
  switch (params.status) {
    case "completed":
      return { unified: "stop", raw: "completed" };
    case "failed":
      return { unified: "error", raw: "failed" };
    default:
      return { unified: "other", raw: params.status };
  }
}

function toToolOutputString(output: LanguageModelV3ToolResultPart["output"]): string {
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

function isHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function convertPromptToInput(prompt: LanguageModelV3Prompt): OpenClawInputItem[] {
  const input: OpenClawInputItem[] = [];

  for (const message of prompt) {
    if (message.role === "system" || message.role === "developer") {
      const text = message.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n");
      if (text) {
        input.push({ type: "message", role: message.role, content: text });
      }
      continue;
    }

    if (message.role === "user" || message.role === "assistant") {
      const contentParts: OpenClawInputPart[] = [];
      const assistant = message.role === "assistant";

      for (const part of message.content) {
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
      for (const part of message.content) {
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

function mapTools(tools: LanguageModelV3CallOptions["tools"], warnings: SharedV3Warning[]) {
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

function mapToolChoice(toolChoice: LanguageModelV3CallOptions["toolChoice"]) {
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

function buildHeaders(
  baseHeaders: Record<string, string>,
  headers: SharedV3Headers | undefined,
  providerOptions: OpenClawProviderOptions | undefined,
): Record<string, string> {
  const extra: Record<string, string> = {};
  if (providerOptions?.sessionKey) {
    extra["x-openclaw-session-key"] = providerOptions.sessionKey;
  }
  if (providerOptions?.agentId) {
    extra["x-openclaw-agent-id"] = providerOptions.agentId;
  }
  return combineHeaders(baseHeaders, headers, extra);
}

export class OpenClawResponsesLanguageModel implements LanguageModelV3 {
  specificationVersion = "v3" as const;
  supportedUrls = {
    "image/*": [/^https?:\/\/.*$/],
    "application/pdf": [/^https?:\/\/.*$/],
  } as const;

  private readonly modelId: string;
  private readonly config: OpenClawResponsesConfig;

  constructor(modelId: string, config: OpenClawResponsesConfig) {
    this.modelId = modelId;
    this.config = config;
  }

  get provider() {
    return this.config.providerName;
  }

  private async getArgs(options: LanguageModelV3CallOptions) {
    const warnings: SharedV3Warning[] = [];

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

  async doGenerate(options: LanguageModelV3CallOptions) {
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

    const content: LanguageModelV3Content[] = [];
    let sawToolCall = false;

    for (const item of response.output) {
      if (item.type === "message") {
        for (const part of item.content) {
          content.push({
            type: "text",
            text: part.text,
            providerMetadata: {
              [this.config.providerName]: { itemId: item.id },
            } as SharedV3ProviderMetadata,
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
          } as SharedV3ProviderMetadata,
        });
      } else if (item.type === "reasoning") {
        const text = item.summary ?? item.content;
        if (text) {
          content.push({
            type: "reasoning",
            text,
            providerMetadata: {
              [this.config.providerName]: { itemId: item.id },
            } as SharedV3ProviderMetadata,
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

  async doStream(options: LanguageModelV3CallOptions) {
    const { body, warnings, providerOptions } = await this.getArgs(options);
    const url = `${this.config.baseURL.replace(/\/$/, "")}/responses`;

    const { responseHeaders, value: response } = await postJsonToApi({
      url,
      headers: buildHeaders(this.config.headers(), options.headers, providerOptions),
      body: { ...body, stream: true },
      failedResponseHandler: openclawFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(openclawStreamEventSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let responseId: string | undefined;
    let responseModel: string | undefined;
    let sawToolCall = false;
    let usage: LanguageModelV3Usage | undefined;
    let finishReason: LanguageModelV3FinishReason | undefined;
    let activeTextId: string | null = null;

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
                responseModel = event.response.model;
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
                      [this.config.providerName]: { itemId: item.id },
                    } as SharedV3ProviderMetadata,
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
                ? ({ [this.config.providerName]: { responseId } } as SharedV3ProviderMetadata)
                : undefined,
            });
          },
        }),
      ),
    };
  }
}
