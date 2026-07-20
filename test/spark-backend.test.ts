import assert from "node:assert/strict";
import test from "node:test";
import {
  clampGPUCountingBucketCount,
  getSparkWebGPUProfile,
} from "../src/backends/webgpu/profiles";

test("mobile WebGPU profile uses conservative defaults", () => {
  const profile = getSparkWebGPUProfile({ mobile: true });
  assert.equal(profile.maxPixelRadius, 128);
  assert.equal(profile.maxSplats, 1_750_000);
  assert.equal(profile.maxPixelRatio, 1.25);
  assert.equal(profile.gpuSortAlgorithm, "counting");
  assert.equal(profile.gpuSortBucketCount, 65_536);
});

test("WebGPU profile accepts platform-specific overrides", () => {
  const profile = getSparkWebGPUProfile({
    mobile: false,
    desktopProfile: { maxSplats: 2_000_000, maxPixelRatio: 1.5 },
  });
  assert.equal(profile.maxSplats, 2_000_000);
  assert.equal(profile.maxPixelRatio, 1.5);
  assert.equal(profile.gpuSortAlgorithm, "adaptive");
});

test("desktop WebGPU profile supports six million GPU-sorted splats", () => {
  const profile = getSparkWebGPUProfile({ mobile: false });
  assert.equal(profile.maxSplats, 6_000_000);
});

test("counting sort keeps at least 16-bit depth precision", () => {
  assert.equal(clampGPUCountingBucketCount(1_024), 65_536);
  assert.equal(clampGPUCountingBucketCount(32_768), 65_536);
  assert.equal(clampGPUCountingBucketCount(65_536), 65_536);
  assert.equal(clampGPUCountingBucketCount(131_072), 131_072);
});
