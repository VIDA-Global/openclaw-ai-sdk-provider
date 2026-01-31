import type { ProviderV3 } from "@ai-sdk/provider";

import { OpenClawResponsesLanguageModel } from "./openclaw-responses-language-model.js";

export type OpenClawProviderSettings = {
  apiKey?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  generateId?: () => string;
  providerName?: string;
};

const DEFAULT_BASE_URL = "http://localhost:18789/v1";

export function createOpenClaw(settings: OpenClawProviderSettings = {}): ProviderV3 & ((modelId: string) => OpenClawResponsesLanguageModel) {
  const apiKey =
    settings.apiKey ??
    process.env.OPENCLAW_API_KEY ??
    process.env.OPENCLAW_TOKEN ??
    process.env.CLAWDBOT_TOKEN ??
    undefined;

  const baseURL =
    settings.baseURL ??
    process.env.OPENCLAW_BASE_URL ??
    process.env.CLAWDBOT_BASE_URL ??
    DEFAULT_BASE_URL;

  const providerName = settings.providerName ?? "openclaw.responses";

  const headers = () => {
    const out: Record<string, string> = {
      ...(settings.headers ?? {}),
    };
    if (apiKey) {
      out.Authorization = `Bearer ${apiKey}`;
    }
    return out;
  };

  const config = {
    baseURL,
    headers,
    fetch: settings.fetch,
    generateId: settings.generateId,
    providerName,
  };

  const provider = ((modelId: string) => new OpenClawResponsesLanguageModel(modelId, config)) as ProviderV3 &
    ((modelId: string) => OpenClawResponsesLanguageModel);

  provider.languageModel = provider;
  provider.provider = providerName;

  return provider;
}
