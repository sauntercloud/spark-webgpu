import assert from "node:assert/strict";
import test from "node:test";
import {
  getWebGPUCapabilityReport,
  isWebGPUAvailable,
  isWebGPUSupported,
} from "../src/backends/webgpu/capabilities";

test("WebGPU capability checks are safe without navigator.gpu", async () => {
  assert.equal(isWebGPUAvailable(), false);
  assert.equal(await isWebGPUSupported(), false);
  assert.deepEqual(await getWebGPUCapabilityReport(), {
    available: false,
    adapter: null,
    reason: "api-unavailable",
  });
});
