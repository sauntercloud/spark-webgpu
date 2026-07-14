import type * as THREE from "three";
import type WebGPURenderer from "three/src/renderers/webgpu/WebGPURenderer.js";
import type { PackedSplats } from "../../PackedSplats";
import { isWebGPUAvailable } from "./capabilities";
import { WebGPUWGSLSplatMesh } from "./wgslSplats";

export type SparkWebGPURendererParameters = ConstructorParameters<
  typeof WebGPURenderer
>[0];

export class SparkWebGPUBackend {
  readonly kind = "webgpu";
  readonly renderer: WebGPURenderer;
  isReady = false;

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

    const { WebGPURenderer } = await import("three/webgpu");
    const renderer = new WebGPURenderer(parameters);
    await renderer.init();

    const backend = new SparkWebGPUBackend({ renderer });
    backend.isReady = true;
    return backend;
  }

  createSplatMesh(
    packedSplats: PackedSplats,
    options?: ConstructorParameters<typeof WebGPUWGSLSplatMesh>[1],
  ) {
    return new WebGPUWGSLSplatMesh(packedSplats, options);
  }

  render(scene: THREE.Scene, camera: THREE.Camera) {
    const directObjects: WebGPUWGSLSplatMesh[] = [];
    scene.traverse((object) => {
      if (object instanceof WebGPUWGSLSplatMesh) {
        directObjects.push(object);
      }
    });
    if (directObjects.length > 0) {
      directObjects.forEach((object, index) => {
        object.renderWebGPUDirect(this.renderer, camera, {
          clear: index === 0,
        });
      });
      return true;
    }
    this.renderer.render(scene, camera);
    return true;
  }
}
