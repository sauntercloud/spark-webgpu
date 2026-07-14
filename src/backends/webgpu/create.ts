import {
  SparkWebGPUBackend,
  type SparkWebGPURendererParameters,
} from "./SparkWebGPUBackend";

export async function createSparkWebGPUBackend({
  parameters,
}: {
  parameters?: SparkWebGPURendererParameters;
} = {}): Promise<SparkWebGPUBackend> {
  return SparkWebGPUBackend.create({ parameters });
}
