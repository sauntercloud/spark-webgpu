import type * as THREE from "three";
import type WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import type { PackedSplats } from "../../PackedSplats";
import { getWebGPUCapabilityReport, isWebGPUAvailable } from "./capabilities";
import { WebGPUWGSLSplatMesh } from "./wgslSplats";

export type SparkWebGPURendererParameters = ConstructorParameters<
  typeof WebGPURenderer
>[0];

export class SparkWebGPUBackend {
  readonly kind = "webgpu";
  readonly renderer: WebGPURenderer;
  isReady = false;
  deviceLostInfo: GPUDeviceLostInfo | null = null;

  private constructor({ renderer }: { renderer: WebGPURenderer }) {
    this.renderer = renderer;
  }

  static async create({
    parameters,
  }: {
    parameters?: SparkWebGPURendererParameters;
  } = {}): Promise<SparkWebGPUBackend> {
    if (!isWebGPUAvailable()) {
      throw new Error(
        "SparkRenderer WebGPU backend requested but WebGPU is unavailable.",
      );
    }

    const capability = await getWebGPUCapabilityReport();
    if (!capability.available) {
      throw new Error(
        "SparkRenderer WebGPU backend requested but no WebGPU adapter is available.",
      );
    }

    const { WebGPURenderer } = await import("three/webgpu");
    const renderer = new WebGPURenderer(parameters);
    await renderer.init();

    const rendererBackend = renderer.backend as { device?: GPUDevice };
    if (!rendererBackend.device) {
      renderer.dispose();
      throw new Error(
        "SparkRenderer requires a native WebGPU device; WebGL fallback is unsupported.",
      );
    }

    const backend = new SparkWebGPUBackend({ renderer });
    backend.isReady = true;
    void rendererBackend.device.lost.then((info) => {
      backend.isReady = false;
      backend.deviceLostInfo = info;
      console.error(`SparkRenderer WebGPU device lost: ${info.message}`);
    });
    return backend;
  }

  createSplatMesh(
    packedSplats: PackedSplats,
    options?: ConstructorParameters<typeof WebGPUWGSLSplatMesh>[1],
  ) {
    return new WebGPUWGSLSplatMesh(packedSplats, options);
  }

  render(scene: THREE.Scene, camera: THREE.Camera) {
    if (!this.isReady) return false;
    const directObjects: WebGPUWGSLSplatMesh[] = [];
    scene.traverse((object) => {
      if (object instanceof WebGPUWGSLSplatMesh) {
        directObjects.push(object);
      }
    });
    this.renderer.render(scene, camera);
    for (const object of directObjects) {
      object.renderWebGPUDirect(this.renderer, camera, { clear: false });
    }
    return true;
  }

  dispose() {
    this.isReady = false;
    this.renderer.dispose();
  }
}
