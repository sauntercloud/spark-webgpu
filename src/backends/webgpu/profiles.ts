import type { CreateWebGPUWGSLSplatOptions } from "./wgslSplats";

export interface SparkWebGPUProfileOptions {
  maxPixelRadius?: number;
  maxStdDev?: number;
  maxSplats?: number;
  maxPixelRatio?: number;
  gpuSortBucketCount?: number;
  gpuSortAlgorithm?: CreateWebGPUWGSLSplatOptions["gpuSortAlgorithm"];
}

const defaultMobileProfile: Required<SparkWebGPUProfileOptions> = {
  maxPixelRadius: 128,
  maxStdDev: 2.2,
  maxSplats: 1_750_000,
  maxPixelRatio: 1.25,
  gpuSortBucketCount: 65_536,
  gpuSortAlgorithm: "counting",
};

const defaultDesktopProfile: Required<SparkWebGPUProfileOptions> = {
  maxPixelRadius: 512,
  maxStdDev: Math.sqrt(8),
  maxSplats: 6_000_000,
  maxPixelRatio: 2,
  gpuSortBucketCount: 131_072,
  gpuSortAlgorithm: "adaptive",
};

export function getSparkWebGPUProfile({
  mobile,
  mobileProfile,
  desktopProfile,
}: {
  mobile: boolean;
  mobileProfile?: SparkWebGPUProfileOptions;
  desktopProfile?: SparkWebGPUProfileOptions;
}): Required<SparkWebGPUProfileOptions> {
  return {
    ...(mobile ? defaultMobileProfile : defaultDesktopProfile),
    ...(mobile ? mobileProfile : desktopProfile),
  };
}

export function clampGPUCountingBucketCount(value: number) {
  if (!Number.isFinite(value)) return 131_072;
  return Math.floor(value) <= 65_536 ? 65_536 : 131_072;
}
