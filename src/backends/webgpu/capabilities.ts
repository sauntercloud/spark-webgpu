export function isWebGPUAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

export interface WebGPUCapabilityReport {
  available: boolean;
  adapter: GPUAdapter | null;
  reason?: "api-unavailable" | "adapter-unavailable";
}

export async function getWebGPUCapabilityReport({
  powerPreference = "low-power",
  ...options
}: GPURequestAdapterOptions = {}): Promise<WebGPUCapabilityReport> {
  if (!isWebGPUAvailable()) {
    return { available: false, adapter: null, reason: "api-unavailable" };
  }
  const adapter = await navigator.gpu.requestAdapter({
    ...options,
    powerPreference,
  });
  return adapter
    ? { available: true, adapter }
    : { available: false, adapter: null, reason: "adapter-unavailable" };
}

export async function isWebGPUSupported(
  options?: GPURequestAdapterOptions,
): Promise<boolean> {
  return (await getWebGPUCapabilityReport(options)).available;
}
