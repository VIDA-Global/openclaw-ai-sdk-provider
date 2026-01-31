export declare class OpenClawResponsesLanguageModel {
  specificationVersion: "v3";
  supportedUrls: {
    "image/*": RegExp[];
    "application/pdf": RegExp[];
  };
  constructor(
    modelId: string,
    config: {
      baseURL: string;
      headers: () => Record<string, string>;
      fetch?: typeof fetch;
      generateId?: () => string;
      providerName: string;
    },
  );
  get provider(): string;
  doGenerate(options: any): Promise<any>;
  doStream(options: any): Promise<any>;
}
