import * as THREE from "three";
import type WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import type { PackedSplatsOptions } from "./PackedSplats";
import { PackedSplats } from "./PackedSplats";
import { SparkRenderer, type SparkRendererOptions } from "./SparkRenderer";
import { SplatMesh, type SplatMeshOptions } from "./SplatMesh";
import {
  SparkWebGPUBackend,
  type SparkWebGPURendererParameters,
} from "./backends/webgpu/SparkWebGPUBackend";
import {
  type SparkWebGPUProfileOptions,
  getSparkWebGPUProfile,
} from "./backends/webgpu/profiles";
import {
  type CreateWebGPUWGSLSplatOptions,
  WebGPUWGSLSplatMesh,
} from "./backends/webgpu/wgslSplats";
import { isMobile } from "./utils";

export {
  getSparkWebGPUProfile,
  type SparkWebGPUProfileOptions,
} from "./backends/webgpu/profiles";

export type SparkBackendPreference = "auto" | "webgpu" | "webgl2";
export type SparkBackendKind = Exclude<SparkBackendPreference, "auto">;
export type SparkRendererInstance = THREE.WebGLRenderer | WebGPURenderer;
export type SparkSplatMesh = SplatMesh | WebGPUWGSLSplatMesh;

export interface CreateSparkRendererOptions {
  backend?: SparkBackendPreference;
  canvas?: HTMLCanvasElement | OffscreenCanvas;
  webgl2?: Omit<THREE.WebGLRendererParameters, "canvas">;
  webgpu?: SparkWebGPURendererParameters;
  spark?: Omit<SparkRendererOptions, "renderer">;
  webgpuSplat?: CreateWebGPUWGSLSplatOptions;
  mobileProfile?: SparkWebGPUProfileOptions;
  desktopProfile?: SparkWebGPUProfileOptions;
  onFallback?: (error: unknown) => void;
}

export interface CreateSparkSplatOptions
  extends Pick<
    SplatMeshOptions,
    | "url"
    | "fileBytes"
    | "fileType"
    | "fileName"
    | "stream"
    | "streamLength"
    | "packedSplats"
    | "maxSplats"
    | "onProgress"
    | "splatEncoding"
  > {
  webgpu?: CreateWebGPUWGSLSplatOptions;
  webgl2?: Omit<
    SplatMeshOptions,
    | "url"
    | "fileBytes"
    | "fileType"
    | "fileName"
    | "stream"
    | "streamLength"
    | "packedSplats"
    | "maxSplats"
    | "onProgress"
    | "splatEncoding"
  >;
}

export class SparkBackend {
  readonly renderer: SparkRendererInstance;
  readonly backend: SparkBackendKind;
  readonly sparkRenderer: SparkRenderer | null;
  readonly webgpuBackend: SparkWebGPUBackend | null;
  readonly profile: Required<SparkWebGPUProfileOptions>;
  private readonly webgpuSplatOptions: CreateWebGPUWGSLSplatOptions;
  private readonly splats = new Set<SparkSplatMesh>();
  private readonly ownedWebGPUPackedSplats = new Set<PackedSplats>();

  private constructor({
    renderer,
    backend,
    sparkRenderer,
    webgpuBackend,
    profile,
    webgpuSplatOptions,
  }: {
    renderer: SparkRendererInstance;
    backend: SparkBackendKind;
    sparkRenderer: SparkRenderer | null;
    webgpuBackend: SparkWebGPUBackend | null;
    profile: Required<SparkWebGPUProfileOptions>;
    webgpuSplatOptions: CreateWebGPUWGSLSplatOptions;
  }) {
    this.renderer = renderer;
    this.backend = backend;
    this.sparkRenderer = sparkRenderer;
    this.webgpuBackend = webgpuBackend;
    this.profile = profile;
    this.webgpuSplatOptions = webgpuSplatOptions;
  }

  static async create(
    options: CreateSparkRendererOptions = {},
  ): Promise<SparkBackend> {
    const preference = options.backend ?? "auto";
    const mobile = typeof navigator !== "undefined" && isMobile();
    const profile = getSparkWebGPUProfile({
      mobile,
      mobileProfile: options.mobileProfile,
      desktopProfile: options.desktopProfile,
    });

    if (preference !== "webgl2") {
      try {
        const webgpuBackend = await SparkWebGPUBackend.create({
          parameters: {
            antialias: false,
            ...(options.canvas ? { canvas: options.canvas } : {}),
            ...options.webgpu,
          },
        });
        return new SparkBackend({
          renderer: webgpuBackend.renderer,
          backend: "webgpu",
          sparkRenderer: null,
          webgpuBackend,
          profile,
          webgpuSplatOptions: options.webgpuSplat ?? {},
        });
      } catch (error) {
        if (preference === "webgpu") throw error;
        options.onFallback?.(error);
      }
    }

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      ...(options.canvas
        ? { canvas: options.canvas as HTMLCanvasElement }
        : {}),
      ...options.webgl2,
    });
    const sparkRenderer = new SparkRenderer({
      renderer,
      ...options.spark,
    });
    return new SparkBackend({
      renderer,
      backend: "webgl2",
      sparkRenderer,
      webgpuBackend: null,
      profile,
      webgpuSplatOptions: options.webgpuSplat ?? {},
    });
  }

  async createSplatMesh(
    options: CreateSparkSplatOptions = {},
  ): Promise<SparkSplatMesh> {
    if (this.backend === "webgl2") {
      const mesh = new SplatMesh({ ...options, ...options.webgl2 });
      await mesh.initialized;
      this.splats.add(mesh);
      return mesh;
    }

    const packedSplats =
      options.packedSplats ?? new PackedSplats(toPackedSplatsOptions(options));
    await packedSplats.initialized;
    const webgpuOptions = mergeDefined<CreateWebGPUWGSLSplatOptions>(
      {
        sort: true,
        sortGPU: true,
        gpuSortRadix: true,
        maxPixelRadius: this.profile.maxPixelRadius,
        maxStdDev: this.profile.maxStdDev,
        gpuSortMaxSplats: this.profile.maxSplats,
        gpuSortBucketCount: this.profile.gpuSortBucketCount,
        gpuSortAlgorithm: this.profile.gpuSortAlgorithm,
      },
      this.webgpuSplatOptions,
      options.webgpu,
    );
    const mesh = this.webgpuBackend?.createSplatMesh(
      packedSplats,
      webgpuOptions,
    );
    if (!mesh) throw new Error("Spark WebGPU backend is unavailable");
    if (!options.packedSplats) {
      this.ownedWebGPUPackedSplats.add(packedSplats);
    }
    this.splats.add(mesh);
    return mesh;
  }

  render(scene: THREE.Scene, camera: THREE.Camera): boolean {
    if (this.backend === "webgpu") {
      return this.webgpuBackend?.render(scene, camera) ?? false;
    }
    if (this.sparkRenderer && this.sparkRenderer.parent !== scene) {
      scene.add(this.sparkRenderer);
    }
    this.renderer.render(scene, camera);
    return true;
  }

  setSize(width: number, height: number, updateStyle?: boolean) {
    this.renderer.setSize(width, height, updateStyle);
  }

  setPixelRatio(pixelRatio = globalThis.devicePixelRatio ?? 1) {
    this.renderer.setPixelRatio(
      Math.min(pixelRatio, this.profile.maxPixelRatio),
    );
  }

  disposeSplatMesh(splat: SparkSplatMesh) {
    if (!this.splats.delete(splat)) return;
    splat.dispose();
    if (
      splat instanceof WebGPUWGSLSplatMesh &&
      this.ownedWebGPUPackedSplats.delete(splat.packedSplats)
    ) {
      splat.packedSplats.dispose();
    }
  }

  dispose() {
    for (const splat of [...this.splats]) this.disposeSplatMesh(splat);
    this.sparkRenderer?.dispose();
    if (this.webgpuBackend) this.webgpuBackend.dispose();
    else this.renderer.dispose();
  }
}

export function createSparkRenderer(
  options?: CreateSparkRendererOptions,
): Promise<SparkBackend> {
  return SparkBackend.create(options);
}

function toPackedSplatsOptions(
  options: CreateSparkSplatOptions,
): PackedSplatsOptions {
  return {
    url: options.url,
    fileBytes: options.fileBytes,
    fileType: options.fileType,
    fileName: options.fileName,
    stream: options.stream,
    streamLength: options.streamLength,
    maxSplats: options.maxSplats,
    onProgress: options.onProgress,
    splatEncoding: options.splatEncoding,
  };
}

function mergeDefined<T extends object>(
  ...sources: Array<Partial<T> | undefined>
): T {
  const result: Partial<T> = {};
  for (const source of sources) {
    if (!source) continue;
    for (const [key, value] of Object.entries(source)) {
      if (value !== undefined) {
        (result as Record<string, unknown>)[key] = value;
      }
    }
  }
  return result as T;
}
