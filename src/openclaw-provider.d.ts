export type OpenClawProviderSettings = {
  apiKey?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  fetch?: typeof fetch;
  generateId?: () => string;
  providerName?: string;
};

export declare function createOpenClaw(
  settings?: OpenClawProviderSettings,
): ((modelId: string) => any) & {
  languageModel: (modelId: string) => any;
  provider: string;
};
