import * as THREE from "three";
import type { PackedSplats } from "../../PackedSplats";

export interface CreateWebGPUWGSLSplatOptions {
  sizeScale?: number;
  opacityScale?: number;
  minAlpha?: number;
  maxPixelRadius?: number;
  maxStdDev?: number;
  blurAmount?: number;
  preBlurAmount?: number;
  falloff?: number;
  alphaBias?: number;
  alphaRadiusScale?: number;
  highPrecisionProjected?: boolean;
  focalAdjustment?: number;
  cull?: boolean;
  renderEpsilon?: number;
  sort?: boolean;
  sortRadial?: boolean;
  onDirty?: () => void;
  renderSortedCamera?: boolean;
  sortGPU?: boolean;
  gpuSortMaxSplats?: number;
  gpuSortAlgorithm?: "bitonic" | "radix" | "bucket" | "counting" | "adaptive";
  gpuSortBucketBits?: number;
  gpuSortBucketCount?: number;
  gpuSortRadix?: boolean;
  gpuSortRadixUnsafe?: boolean;
  gpuSortDebugOnce?: boolean;
  gpuSortTimestamp?: boolean;
}

interface WebGPUSplatSource {
  getNumSplats(): number;
  forEachSplat(
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ): void;
}

interface WebGPURendererLike {
  backend?: unknown;
  getDrawingBufferSize?: (target: THREE.Vector2) => THREE.Vector2;
  getClearColor?: (target: THREE.Color) => {
    r: number;
    g: number;
    b: number;
  };
  getClearAlpha?: () => number;
}

interface WebGPURendererBackendLike {
  device?: GPUDevice;
  context?: GPUCanvasContext;
  getContext?: () => GPUCanvasContext;
}

export class WebGPUWGSLSplatMesh extends THREE.Object3D {
  readonly packedSplats: PackedSplats;
  readonly sizeScale: number;
  readonly opacityScale: number;
  minAlpha: number;
  readonly maxPixelRadius: number;
  readonly maxStdDev: number;
  blurAmount: number;
  readonly preBlurAmount: number;
  readonly falloff: number;
  alphaBias: number;
  alphaRadiusScale: number;
  highPrecisionProjected: boolean;
  readonly focalAdjustment: number;
  readonly cull: boolean;
  readonly renderEpsilon: number;
  sort: boolean;
  sortRadial: boolean;
  sortPending = false;
  onDirty?: () => void;
  renderSortedCamera: boolean;
  sortGPU: boolean;
  gpuSortMaxSplats: number;
  gpuSortAlgorithm: "bitonic" | "radix" | "bucket" | "counting" | "adaptive";
  gpuSortBucketBits: number;
  gpuSortBucketCount: number;
  gpuSortRadix: boolean;
  gpuSortDebugOnce: boolean;
  gpuSortTimestamp: boolean;

  private renderer: DirectWGSLSplatRenderer | null = null;
  private sortData: DirectWGSLSortData | null = null;
  private orderDirty = false;
  private displayCamera: THREE.Camera | null = null;
  private pendingOrder: { indices: Uint32Array; visibleCount: number } | null =
    null;
  private sortRequestCount = 0;
  private sortCompleteCount = 0;
  private sortPublishCount = 0;
  private sortDeferCount = 0;
  private sortDropCount = 0;
  private latestSortRequestSerial = 0;
  private publishedSortRequestSerial = 0;
  private lastSortLatencyMs = 0;
  private maxSortLatencyMs = 0;
  private lastOrderCameraPosition = new THREE.Vector3(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private lastOrderCameraQuaternion = new THREE.Quaternion(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private sortedOrderCameraPosition = new THREE.Vector3(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private sortedOrderCameraQuaternion = new THREE.Quaternion(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private gpuOrderUpdated = false;
  private gpuSortSubmitCount = 0;
  private lastGPUSortFallbackReason = "none";
  private gpuSortDebugReadbackCount = 0;
  private gpuSortDebugReadbackDone = false;
  private gpuSortInFlight = false;
  private adaptiveGPUStableFrames = 0;
  private adaptiveGPULastPrecisePosition = new THREE.Vector3(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private adaptiveGPULastPreciseQuaternion = new THREE.Quaternion(
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  private adaptiveGPUNeedsPreciseSort = false;
  private adaptiveGPULastResolvedAlgorithm = "none";
  private lastCullLatencyMs = 0;
  private maxCullLatencyMs = 0;
  private lastGPUSortQueueLatencyMs = 0;
  private maxGPUSortQueueLatencyMs = 0;
  private deferredGPUSortCamera: THREE.Camera | null = null;
  private deferredGPUSortPosition = new THREE.Vector3();
  private deferredGPUSortQuaternion = new THREE.Quaternion();
  private lastSortMode = "none";

  constructor(
    packedSplats: PackedSplats,
    {
      sizeScale = 1,
      opacityScale = 1,
      minAlpha = 3 / 255,
      maxPixelRadius = 512,
      maxStdDev = Math.sqrt(8),
      blurAmount = 0.3,
      preBlurAmount = 0,
      falloff = 1,
      alphaBias = 5,
      alphaRadiusScale = 1,
      highPrecisionProjected = false,
      focalAdjustment = 1,
      cull = true,
      renderEpsilon = defaultRenderStateMatrixEpsilon,
      sort = false,
      sortRadial = true,
      onDirty,
      renderSortedCamera = false,
      sortGPU = false,
      gpuSortMaxSplats = defaultExperimentalGPUSortMaxSplats,
      gpuSortAlgorithm = "adaptive",
      gpuSortBucketBits = 16,
      gpuSortBucketCount = 131072,
      gpuSortRadix,
      gpuSortRadixUnsafe = false,
      gpuSortDebugOnce = false,
      gpuSortTimestamp = false,
    }: CreateWebGPUWGSLSplatOptions = {},
  ) {
    super();
    this.packedSplats = packedSplats;
    this.sizeScale = sizeScale;
    this.opacityScale = opacityScale;
    this.minAlpha = minAlpha;
    this.maxPixelRadius = maxPixelRadius;
    this.maxStdDev = maxStdDev;
    this.blurAmount = blurAmount;
    this.preBlurAmount = preBlurAmount;
    this.falloff = falloff;
    this.alphaBias = alphaBias;
    this.alphaRadiusScale = alphaRadiusScale;
    this.highPrecisionProjected = highPrecisionProjected;
    this.focalAdjustment = focalAdjustment;
    this.cull = cull;
    this.renderEpsilon = renderEpsilon;
    this.sort = sort;
    this.sortRadial = sortRadial;
    this.onDirty = onDirty;
    this.renderSortedCamera = renderSortedCamera;
    this.sortGPU = sortGPU;
    this.gpuSortMaxSplats = Math.max(1, Math.floor(gpuSortMaxSplats));
    this.gpuSortAlgorithm =
      gpuSortAlgorithm === "radix" ||
      gpuSortAlgorithm === "bucket" ||
      gpuSortAlgorithm === "counting" ||
      gpuSortAlgorithm === "adaptive"
        ? gpuSortAlgorithm
        : "bitonic";
    this.gpuSortBucketBits = clampGPUSortBucketBits(gpuSortBucketBits);
    this.gpuSortBucketCount = clampGPUCountingBucketCount(gpuSortBucketCount);
    this.gpuSortRadix = gpuSortRadix ?? gpuSortRadixUnsafe;
    this.gpuSortDebugOnce = gpuSortDebugOnce;
    this.gpuSortTimestamp = gpuSortTimestamp;
    this.frustumCulled = false;
  }

  updateFrame() {
    return false;
  }

  requestSortUpdate() {
    return false;
  }

  updateSort() {
    return false;
  }

  setRenderOptions({
    minAlpha,
    alphaBias,
    alphaRadiusScale,
    blurAmount,
  }: {
    minAlpha?: number;
    alphaBias?: number;
    alphaRadiusScale?: number;
    blurAmount?: number;
  }) {
    let changed = false;
    if (minAlpha !== undefined && Number.isFinite(minAlpha)) {
      this.minAlpha = Math.max(0, minAlpha);
      changed = true;
    }
    if (alphaBias !== undefined && Number.isFinite(alphaBias)) {
      this.alphaBias = Math.max(0.000001, alphaBias);
      changed = true;
    }
    if (alphaRadiusScale !== undefined && Number.isFinite(alphaRadiusScale)) {
      this.alphaRadiusScale = Math.max(0.000001, alphaRadiusScale);
      changed = true;
    }
    if (blurAmount !== undefined && Number.isFinite(blurAmount)) {
      this.blurAmount = Math.max(0, blurAmount);
      changed = true;
    }
    if (changed) {
      this.renderer?.setRenderOptions({
        minAlpha: this.minAlpha,
        alphaBias: this.alphaBias,
        alphaRadiusScale: this.alphaRadiusScale,
        blurAmount: this.blurAmount,
      });
      this.notifyDirty();
    }
  }

  dispose() {}

  renderWebGPUDirect(
    renderer: WebGPURendererLike,
    camera: THREE.Camera,
    { clear = true }: { clear?: boolean } = {},
  ) {
    const backend = renderer.backend as WebGPURendererBackendLike | undefined;
    const device = backend?.device;
    const context = backend?.context ?? backend?.getContext?.();
    if (!device || !context) {
      console.warn("Spark WebGPU direct renderer missing device/context");
      return false;
    }
    this.updateMatrixWorld();
    camera.updateMatrixWorld();
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    this.renderer ??= new DirectWGSLSplatRenderer(
      device,
      context,
      this.packedSplats,
      {
        sizeScale: this.sizeScale,
        opacityScale: this.opacityScale,
        minAlpha: this.minAlpha,
        maxPixelRadius: this.maxPixelRadius,
        maxStdDev: this.maxStdDev,
        blurAmount: this.blurAmount,
        preBlurAmount: this.preBlurAmount,
        falloff: this.falloff,
        alphaBias: this.alphaBias,
        alphaRadiusScale: this.alphaRadiusScale,
        highPrecisionProjected: this.highPrecisionProjected,
        focalAdjustment: this.focalAdjustment,
        renderEpsilon: this.renderEpsilon,
        gpuSortTimestamp: this.gpuSortTimestamp,
      },
      () => this.notifyDirty(),
    );
    this.updateOrder(camera);
    try {
      let uploadedOrder = this.gpuOrderUpdated;
      this.gpuOrderUpdated = false;
      if (this.orderDirty) {
        this.renderer.updateOrder(
          this.pendingOrder?.indices ?? this.sortData?.indices ?? null,
          this.pendingOrder?.visibleCount ?? this.sortData?.visibleCount,
        );
        this.renderer.invalidateRenderState("order");
        this.pendingOrder = null;
        this.orderDirty = false;
        uploadedOrder = true;
      }
      if (this.sort && !this.displayCamera) {
        this.renderer.clear(renderer);
        return true;
      }
      const renderCamera =
        this.sort && this.renderSortedCamera && this.displayCamera
          ? this.displayCamera
          : camera;
      this.renderer.render(renderer, renderCamera, this, clear, {
        force: uploadedOrder,
      });
      return true;
    } catch (error) {
      console.error("Spark WebGPU direct render failed", error);
      return false;
    }
  }

  get opacity() {
    return this.opacityScale;
  }

  get visibleDrawCount() {
    return (
      this.renderer?.getDrawCount() ??
      this.sortData?.visibleCount ??
      this.packedSplats.getNumSplats()
    );
  }

  getDebugState() {
    return {
      sortPending: this.sortPending,
      hasDisplayCamera: Boolean(this.displayCamera),
      renderSortedCamera: this.renderSortedCamera,
      sortGPU: this.sortGPU,
      lastSortMode: this.lastSortMode,
      gpuSortSubmitCount: this.gpuSortSubmitCount,
      lastGPUSortFallbackReason: this.lastGPUSortFallbackReason,
      gpuSortMaxSplats: this.gpuSortMaxSplats,
      gpuSortAlgorithm: this.gpuSortAlgorithm,
      gpuSortBucketBits: this.gpuSortBucketBits,
      gpuSortBucketCount: this.gpuSortBucketCount,
      adaptiveGPUStableFrames: this.adaptiveGPUStableFrames,
      adaptiveGPULastResolvedAlgorithm: this.adaptiveGPULastResolvedAlgorithm,
      gpuSortRadix: this.gpuSortRadix,
      gpuSortRadixUnsafe: this.gpuSortRadix,
      gpuSortDebugReadbackCount: this.gpuSortDebugReadbackCount,
      gpuSortTimestamp: this.gpuSortTimestamp,
      gpuSortInFlight: this.gpuSortInFlight,
      lastCullLatencyMs: this.lastCullLatencyMs,
      maxCullLatencyMs: this.maxCullLatencyMs,
      lastGPUSortQueueLatencyMs: this.lastGPUSortQueueLatencyMs,
      maxGPUSortQueueLatencyMs: this.maxGPUSortQueueLatencyMs,
      rendererSupportsTimestampQuery:
        this.renderer?.supportsTimestampQuery ?? false,
      lastGPURadixTimings: this.renderer?.lastGPURadixTimings ?? null,
      hasDeferredGPUSort: Boolean(this.deferredGPUSortCamera),
      hasDeferredSort: false,
      orderDirty: this.orderDirty,
      visibleDrawCount: this.visibleDrawCount,
      sortRequestCount: this.sortRequestCount,
      sortCompleteCount: this.sortCompleteCount,
      sortPublishCount: this.sortPublishCount,
      sortDeferCount: this.sortDeferCount,
      sortDropCount: this.sortDropCount,
      latestSortRequestSerial: this.latestSortRequestSerial,
      publishedSortRequestSerial: this.publishedSortRequestSerial,
      lastSortLatencyMs: this.lastSortLatencyMs,
      maxSortLatencyMs: this.maxSortLatencyMs,
      renderer: this.renderer?.getDebugState() ?? null,
    };
  }

  requestDeferredRender() {
    this.renderer?.requestDeferredRender();
  }

  get isInitialized() {
    return this.packedSplats.isInitialized;
  }

  get initialized() {
    return this.packedSplats.initialized.then(() => this);
  }

  private updateOrder(camera: THREE.Camera) {
    if (!this.sort && !this.cull) {
      this.sortData = null;
      this.orderDirty = false;
      return;
    }
    const position = camera.getWorldPosition(scratchOrderPosition);
    const quaternion = camera.getWorldQuaternion(scratchOrderQuaternion);
    const sortedChanged =
      position.distanceTo(this.sortedOrderCameraPosition) > 0.001 ||
      Math.abs(quaternion.dot(this.sortedOrderCameraQuaternion)) < 0.9999;
    const needsAdaptivePreciseSort =
      this.sort &&
      this.sortGPU &&
      this.gpuSortAlgorithm === "adaptive" &&
      this.adaptiveGPUNeedsPreciseSort &&
      (position.distanceTo(this.adaptiveGPULastPrecisePosition) > 0.001 ||
        Math.abs(quaternion.dot(this.adaptiveGPULastPreciseQuaternion)) <
          0.9999);
    if (!sortedChanged && this.sortData && !needsAdaptivePreciseSort) {
      return;
    }
    this.sortData ??= createDirectWGSLSortData(this.packedSplats);
    if (this.sort) {
      if (!this.sortGPU) {
        this.lastSortMode = "none";
        return;
      }
      if (!this.sortDirectGPU(camera, position, quaternion)) {
        this.lastSortMode = "gpu-unavailable";
        console.warn(
          `Spark WebGPU GPU sort unavailable: ${this.lastGPUSortFallbackReason}`,
        );
      }
      return;
    }
    this.sortData.visibleCount = this.sortData.indices.length;
    this.lastOrderCameraPosition.copy(position);
    this.lastOrderCameraQuaternion.copy(quaternion);
    this.sortedOrderCameraPosition.copy(position);
    this.sortedOrderCameraQuaternion.copy(quaternion);
    this.orderDirty = true;
    this.notifyDirty();
  }

  private sortDirectGPU(
    camera: THREE.Camera,
    position: THREE.Vector3,
    quaternion: THREE.Quaternion,
  ) {
    if (!this.sortData || !this.renderer) {
      this.lastGPUSortFallbackReason = !this.sortData
        ? "missing-sort-data"
        : "missing-renderer";
      return false;
    }
    if (this.gpuSortInFlight) {
      this.sortDeferCount++;
      this.deferGPUSortRequest(camera, position, quaternion);
      this.sortPending = true;
      this.lastOrderCameraPosition.copy(position);
      this.lastOrderCameraQuaternion.copy(quaternion);
      return true;
    }
    const requestStart = performance.now();
    const localToView = scratchSortMatrix
      .copy(camera.matrixWorldInverse)
      .multiply(this.matrixWorld);
    const adaptiveSort = this.resolveAdaptiveGPUSort(position, quaternion);
    const sorted = this.renderer.updateOrderGPU(
      this.sortData.positions,
      new Float32Array(localToView.elements),
      this.sortRadial,
      this.gpuSortMaxSplats,
      adaptiveSort.algorithm,
      adaptiveSort.bucketBits,
      adaptiveSort.bucketCount,
      this.gpuSortRadix,
      this.gpuSortDebugOnce && !this.gpuSortDebugReadbackDone,
      this.gpuSortTimestamp,
    );
    if (!sorted.ok) {
      this.lastGPUSortFallbackReason = sorted.reason;
      return false;
    }
    this.lastGPUSortFallbackReason = "none";
    this.gpuSortSubmitCount++;
    this.lastSortMode =
      this.gpuSortAlgorithm === "adaptive"
        ? `gpu:${adaptiveSort.label}`
        : "gpu";
    if (sorted.debugReadback) {
      this.gpuSortDebugReadbackDone = true;
      this.gpuSortDebugReadbackCount++;
      this.logGPUOrderDebug(
        sorted.debugReadback,
        this.packedSplats.getNumSplats(),
      );
    }
    const requestSerial = ++this.latestSortRequestSerial;
    this.sortRequestCount++;
    this.sortCompleteCount++;
    this.sortPublishCount++;
    this.publishedSortRequestSerial = requestSerial;
    this.lastOrderCameraPosition.copy(position);
    this.lastOrderCameraQuaternion.copy(quaternion);
    this.sortedOrderCameraPosition.copy(position);
    this.sortedOrderCameraQuaternion.copy(quaternion);
    if (adaptiveSort.algorithm === "radix") {
      this.adaptiveGPULastPrecisePosition.copy(position);
      this.adaptiveGPULastPreciseQuaternion.copy(quaternion);
      this.adaptiveGPUNeedsPreciseSort = false;
    } else if (
      this.gpuSortAlgorithm === "adaptive" &&
      adaptiveSort.label === "counting131072"
    ) {
      this.adaptiveGPUNeedsPreciseSort = true;
    }
    this.sortData.visibleCount = sorted.activeCount;
    this.displayCamera = cloneCameraState(camera);
    this.gpuOrderUpdated = true;
    this.gpuSortInFlight = true;
    this.sortPending = true;
    sorted.complete.finally(() => {
      const queueLatency = performance.now() - requestStart;
      this.lastGPUSortQueueLatencyMs = queueLatency;
      this.maxGPUSortQueueLatencyMs = Math.max(
        this.maxGPUSortQueueLatencyMs,
        queueLatency,
      );
      this.gpuSortInFlight = false;
      this.sortPending = Boolean(this.deferredGPUSortCamera);
      if (
        this.gpuSortAlgorithm === "adaptive" &&
        this.adaptiveGPUNeedsPreciseSort &&
        this.adaptiveGPUStableFrames >= 3 &&
        !this.deferredGPUSortCamera
      ) {
        this.notifyDirty();
      }
      this.flushDeferredGPUSortRequest();
    });
    this.lastSortLatencyMs = performance.now() - requestStart;
    this.maxSortLatencyMs = Math.max(
      this.maxSortLatencyMs,
      this.lastSortLatencyMs,
    );
    this.renderer.invalidateRenderState("order");
    this.notifyDirty();
    return true;
  }

  private resolveAdaptiveGPUSort(
    position: THREE.Vector3,
    quaternion: THREE.Quaternion,
  ): {
    algorithm: "bitonic" | "radix" | "bucket" | "counting";
    bucketBits: number;
    bucketCount: number;
    label: string;
  } {
    if (this.gpuSortAlgorithm !== "adaptive") {
      this.adaptiveGPULastResolvedAlgorithm = this.gpuSortAlgorithm;
      return {
        algorithm: this.gpuSortAlgorithm,
        bucketBits: this.gpuSortBucketBits,
        bucketCount: this.gpuSortBucketCount,
        label: this.gpuSortAlgorithm,
      };
    }

    const positionDelta = position.distanceTo(this.sortedOrderCameraPosition);
    const quaternionDot = Math.min(
      1,
      Math.abs(quaternion.dot(this.sortedOrderCameraQuaternion)),
    );
    const angleDelta = 2 * Math.acos(quaternionDot);
    const stable = positionDelta < 0.003 && angleDelta < 0.003;
    if (stable) {
      this.adaptiveGPUStableFrames++;
    } else {
      this.adaptiveGPUStableFrames = 0;
    }

    const precisePositionDelta = position.distanceTo(
      this.adaptiveGPULastPrecisePosition,
    );
    const preciseQuaternionDot = Math.min(
      1,
      Math.abs(quaternion.dot(this.adaptiveGPULastPreciseQuaternion)),
    );
    const preciseAngleDelta = 2 * Math.acos(preciseQuaternionDot);
    const preciseCurrent =
      precisePositionDelta < 0.001 && preciseAngleDelta < 0.001;
    if (this.adaptiveGPUStableFrames >= 3 && !preciseCurrent) {
      this.adaptiveGPULastResolvedAlgorithm = "radix";
      return {
        algorithm: "radix",
        bucketBits: 32,
        bucketCount: 65536,
        label: "radix",
      };
    }

    const bucketCount = 131072;
    const label = "counting131072";
    this.adaptiveGPULastResolvedAlgorithm = label;
    return {
      algorithm: "counting",
      bucketBits: 16,
      bucketCount,
      label,
    };
  }

  private logGPUOrderDebug(debugReadback: GPUSortDebugReadback, count: number) {
    void debugReadback.promise
      .then((order) => {
        const sampleCount = Math.min(order.length, 4096);
        const seen = new Set<number>();
        let duplicateCount = 0;
        let outOfRangeCount = 0;
        for (let i = 0; i < sampleCount; i++) {
          const value = order[i];
          if (value >= count) {
            outOfRangeCount++;
          }
          if (seen.has(value)) {
            duplicateCount++;
          }
          seen.add(value);
        }
        console.log("Spark WebGPU GPU sort order debug", {
          sampleCount,
          duplicateCount,
          outOfRangeCount,
          head: Array.from(order.subarray(0, Math.min(32, order.length))),
        });
      })
      .catch((error) => {
        console.warn("Spark WebGPU GPU sort order debug failed", error);
      });
  }

  private deferGPUSortRequest(
    camera: THREE.Camera,
    position: THREE.Vector3,
    quaternion: THREE.Quaternion,
  ) {
    this.deferredGPUSortCamera ??= camera.clone();
    this.deferredGPUSortCamera.matrixWorld.copy(camera.matrixWorld);
    this.deferredGPUSortCamera.matrixWorldInverse.copy(
      camera.matrixWorldInverse,
    );
    this.deferredGPUSortCamera.projectionMatrix.copy(camera.projectionMatrix);
    this.deferredGPUSortCamera.projectionMatrixInverse.copy(
      camera.projectionMatrixInverse,
    );
    this.deferredGPUSortPosition.copy(position);
    this.deferredGPUSortQuaternion.copy(quaternion);
  }

  private flushDeferredGPUSortRequest() {
    if (!this.deferredGPUSortCamera || this.gpuSortInFlight) {
      return;
    }
    const camera = this.deferredGPUSortCamera;
    const position = scratchFlushPosition.copy(this.deferredGPUSortPosition);
    const quaternion = scratchFlushQuaternion.copy(
      this.deferredGPUSortQuaternion,
    );
    this.deferredGPUSortCamera = null;
    this.sortPending = false;
    this.sortDirectGPU(camera, position, quaternion);
  }

  private notifyDirty() {
    this.onDirty?.();
  }
}

interface DirectWGSLSortData {
  positions: Float32Array;
  indices: Uint32Array;
  visibleCount: number;
}

type GPUSortState = {
  paddedCount: number;
  positionsBuffer: GPUBuffer;
  pairsBuffer: GPUBuffer;
  orderBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  passStates: Map<string, GPUSortPassState>;
};

type GPUSortPassState = {
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
};

type GPURadixSortState = {
  count: number;
  blockCount: number;
  blockGroupCount: number;
  positionsBuffer: GPUBuffer;
  pairsA: GPUBuffer;
  pairsB: GPUBuffer;
  orderBuffer: GPUBuffer;
  histogramBuffer: GPUBuffer;
  bucketPrefixBuffer: GPUBuffer;
  prefixBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  passStates: Map<string, GPURadixPassState>;
};

type GPURadixPassState = {
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
};

type GPURadixSortStateResult =
  | { ok: true; state: GPURadixSortState }
  | { ok: false; reason: string };

type GPUCountingSortState = {
  count: number;
  bucketCount: number;
  positionsBuffer: GPUBuffer;
  keysBuffer: GPUBuffer;
  orderBuffer: GPUBuffer;
  histogramBuffer: GPUBuffer;
  offsetsBuffer: GPUBuffer;
  groupTotalsBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
};

type GPUCountingSortStateResult =
  | { ok: true; state: GPUCountingSortState }
  | { ok: false; reason: string };

type GPUSortDebugTarget = {
  readBuffer: GPUBuffer;
};

type GPUSortDebugReadback = {
  promise: Promise<Uint32Array>;
};

type GPUOrderUpdateResult =
  | {
      ok: true;
      activeCount: number;
      complete: Promise<void>;
      debugReadback?: GPUSortDebugReadback;
    }
  | { ok: false; reason: string };

type GPURadixTimingResult = {
  totalMs: number;
  keysMs: number;
  histogramMs: number;
  bucketTotalsMs: number;
  bucketBasesMs: number;
  blockPrefixMs: number;
  scatterMs: number;
  extractMs: number;
  copyMs: number;
  projectMs?: number;
  renderMs?: number;
};

type GPURenderTimingResult = {
  totalMs: number;
  projectMs: number;
  renderMs: number;
};

type GPUTimestampState = {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  nextQuery: number;
  labels: string[];
};

class DirectWGSLSplatRenderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly count: number;
  private readonly chunks: {
    base: number;
    count: number;
    drawCounts: [number, number];
    textureWidth: number;
    textureHeight: number;
    splatTexture: GPUTexture;
    sh1Texture: GPUTexture;
    sh2Texture: GPUTexture;
    sh3Texture: GPUTexture;
    orderTextures: [GPUTexture, GPUTexture];
    projectedBuffer: GPUBuffer;
    projectedColorOpacityBuffer: GPUBuffer;
    projectedCenterOpacityBuffer: GPUBuffer;
    projectedAxesBuffer: GPUBuffer;
    identityOrderData: Uint32Array;
    uniformBuffer: GPUBuffer;
    projectBindGroups: [GPUBindGroup, GPUBindGroup];
    renderBindGroup: GPUBindGroup;
    gpuSort?: GPUSortState;
    gpuRadixSort?: GPURadixSortState;
    gpuCountingSort?: GPUCountingSortState;
  }[];
  private readonly projectPipeline: GPUComputePipeline;
  private readonly gpuSortKeyPipeline: GPUComputePipeline;
  private readonly gpuSortPassPipeline: GPUComputePipeline;
  private readonly gpuSortExtractPipeline: GPUComputePipeline;
  private readonly gpuSortBindGroupLayout: GPUBindGroupLayout;
  private readonly gpuRadixBindGroupLayout: GPUBindGroupLayout;
  private readonly gpuRadixKeyPipeline: GPUComputePipeline;
  private readonly gpuRadixHistogramPipeline: GPUComputePipeline;
  private readonly gpuRadixBucketTotalPipeline: GPUComputePipeline;
  private readonly gpuRadixBucketBasePipeline: GPUComputePipeline;
  private readonly gpuRadixBlockPrefixPipeline: GPUComputePipeline;
  private readonly gpuRadixScatterPipeline: GPUComputePipeline;
  private readonly gpuRadixExtractPipeline: GPUComputePipeline;
  private readonly gpuCountingBindGroupLayout: GPUBindGroupLayout;
  private readonly gpuCountingKeyHistogramPipeline: GPUComputePipeline;
  private readonly gpuCountingBlockPrefixPipeline: GPUComputePipeline;
  private readonly gpuCountingGroupPrefixPipeline: GPUComputePipeline;
  private readonly gpuCountingAddGroupBasePipeline: GPUComputePipeline;
  private readonly gpuCountingScatterPipeline: GPUComputePipeline;
  private readonly renderPipeline: GPURenderPipeline;
  private readonly quadVertexBuffer: GPUBuffer;
  private readonly quadIndexBuffer: GPUBuffer;
  private readonly drawingBufferSize = new THREE.Vector2(1, 1);
  private readonly localToView = new THREE.Matrix4();
  private readonly clearColor = new THREE.Color();
  private readonly cameraPosition = new THREE.Vector3();
  private readonly localCameraPosition = new THREE.Vector3();
  private readonly cameraDirection = new THREE.Vector3();
  private readonly lastCameraPosition = new THREE.Vector3();
  private readonly lastCameraDirection = new THREE.Vector3(0, 0, -1);
  private readonly lastProjectionState = new Float32Array(18);
  private readonly projectionState = new Float32Array(18);
  private readonly uniformData = new Float32Array(56);
  private readonly gpuSortUniformBufferData = new ArrayBuffer(
    24 * Float32Array.BYTES_PER_ELEMENT,
  );
  private readonly gpuSortUniformFloatData = new Float32Array(
    this.gpuSortUniformBufferData,
  );
  private readonly gpuSortUniformUintData = new Uint32Array(
    this.gpuSortUniformBufferData,
  );
  private readonly gpuRadixUniformBufferData = new ArrayBuffer(
    24 * Float32Array.BYTES_PER_ELEMENT,
  );
  private readonly gpuRadixUniformFloatData = new Float32Array(
    this.gpuRadixUniformBufferData,
  );
  private readonly gpuRadixUniformUintData = new Uint32Array(
    this.gpuRadixUniformBufferData,
  );
  private readonly gpuCountingUniformBufferData = new ArrayBuffer(
    24 * Float32Array.BYTES_PER_ELEMENT,
  );
  private readonly gpuCountingUniformFloatData = new Float32Array(
    this.gpuCountingUniformBufferData,
  );
  private readonly gpuCountingUniformUintData = new Uint32Array(
    this.gpuCountingUniformBufferData,
  );
  private readonly lastRenderState = new Float32Array(56);
  private hasSubmittedFrame = false;
  private inFlightFrameCount = 0;
  private submittedFrameCount = 0;
  private skippedStableCount = 0;
  private skippedBusyCount = 0;
  private skippedBusyRenderCount = 0;
  private skippedBusyClearCount = 0;
  private clearFrameCount = 0;
  private deferredRenderRequested = false;
  private lastSubmittedPass = "none";
  private lastSubmittedProjectDispatches = 0;
  private lastSubmittedDrawCalls = 0;
  private lastSubmittedDrawSplats = 0;
  private lastBusyPass = "none";
  private lastBusyDirtyReason = "none";
  private frontOrderSlot: 0 | 1 = 0;
  readonly supportsTimestampQuery: boolean;
  private timestampState: GPUTimestampState | null = null;
  private timestampReadPending = false;
  lastGPURadixTimings: GPURadixTimingResult | null = null;
  lastGPURenderTimings: GPURenderTimingResult | null = null;
  lastDirtyReason = "initial";
  lastMatrixDelta = Number.POSITIVE_INFINITY;
  lastPositionDelta = Number.POSITIVE_INFINITY;
  lastDirectionAngle = Number.POSITIVE_INFINITY;
  lastProjectionDelta = Number.POSITIVE_INFINITY;
  lastScalarDelta = Number.POSITIVE_INFINITY;
  lastDirtyIndex = -1;

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    private readonly packedSplats: PackedSplats,
    private options: Required<
      Pick<
        CreateWebGPUWGSLSplatOptions,
        | "sizeScale"
        | "opacityScale"
        | "minAlpha"
        | "maxPixelRadius"
        | "maxStdDev"
        | "blurAmount"
        | "preBlurAmount"
        | "falloff"
        | "alphaBias"
        | "alphaRadiusScale"
        | "highPrecisionProjected"
        | "focalAdjustment"
        | "renderEpsilon"
        | "gpuSortTimestamp"
      >
    >,
    private readonly onReadyForDeferredRender?: () => void,
  ) {
    this.device = device;
    this.context = context;
    this.count = packedSplats.getNumSplats();
    this.supportsTimestampQuery =
      device.features?.has?.("timestamp-query") ?? false;
    const renderBindGroupLayout = device.createBindGroupLayout({
      label: "spark_wgsl_render_bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    const projectBindGroupLayout = device.createBindGroupLayout({
      label: "spark_wgsl_project_bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "uint", viewDimension: "2d" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "uint", viewDimension: "2d" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "uint" },
        },
        {
          binding: 8,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "uint" },
        },
        {
          binding: 9,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "uint" },
        },
      ],
    });
    this.projectPipeline = device.createComputePipeline({
      label: "spark_wgsl_project_pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [projectBindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({ code: projectWGSL }),
        entryPoint: "cs_main",
      },
    });
    this.gpuSortBindGroupLayout = device.createBindGroupLayout({
      label: "spark_wgsl_gpu_sort_bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });
    const gpuSortModule = device.createShaderModule({ code: gpuSortWGSL });
    const gpuSortLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.gpuSortBindGroupLayout],
    });
    this.gpuSortKeyPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_sort_key_pipeline",
      layout: gpuSortLayout,
      compute: {
        module: gpuSortModule,
        entryPoint: "make_keys",
      },
    });
    this.gpuSortPassPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_sort_pass_pipeline",
      layout: gpuSortLayout,
      compute: {
        module: gpuSortModule,
        entryPoint: "bitonic_pass",
      },
    });
    this.gpuSortExtractPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_sort_extract_pipeline",
      layout: gpuSortLayout,
      compute: {
        module: gpuSortModule,
        entryPoint: "extract_order",
      },
    });
    this.gpuRadixBindGroupLayout = device.createBindGroupLayout({
      label: "spark_wgsl_gpu_radix_sort_bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 7,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const gpuRadixModule = device.createShaderModule({
      code: gpuRadixSortWGSL,
    });
    const gpuRadixLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.gpuRadixBindGroupLayout],
    });
    this.gpuRadixKeyPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_key_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "make_keys",
      },
    });
    this.gpuRadixHistogramPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_histogram_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "histogram",
      },
    });
    this.gpuRadixBucketTotalPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_bucket_total_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "bucket_totals",
      },
    });
    this.gpuRadixBucketBasePipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_bucket_base_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "bucket_bases",
      },
    });
    this.gpuRadixBlockPrefixPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_block_prefix_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "block_prefix",
      },
    });
    this.gpuRadixScatterPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_scatter_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "scatter",
      },
    });
    this.gpuRadixExtractPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_radix_extract_pipeline",
      layout: gpuRadixLayout,
      compute: {
        module: gpuRadixModule,
        entryPoint: "extract_order",
      },
    });
    this.gpuCountingBindGroupLayout = device.createBindGroupLayout({
      label: "spark_wgsl_gpu_counting_sort_bgl",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const gpuCountingModule = device.createShaderModule({
      code: gpuCountingSortWGSL,
    });
    const gpuCountingLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.gpuCountingBindGroupLayout],
    });
    this.gpuCountingKeyHistogramPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_counting_key_histogram_pipeline",
      layout: gpuCountingLayout,
      compute: {
        module: gpuCountingModule,
        entryPoint: "key_histogram",
      },
    });
    this.gpuCountingBlockPrefixPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_counting_block_prefix_pipeline",
      layout: gpuCountingLayout,
      compute: {
        module: gpuCountingModule,
        entryPoint: "block_prefix",
      },
    });
    this.gpuCountingGroupPrefixPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_counting_group_prefix_pipeline",
      layout: gpuCountingLayout,
      compute: {
        module: gpuCountingModule,
        entryPoint: "group_prefix",
      },
    });
    this.gpuCountingAddGroupBasePipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_counting_add_group_base_pipeline",
      layout: gpuCountingLayout,
      compute: {
        module: gpuCountingModule,
        entryPoint: "add_group_base",
      },
    });
    this.gpuCountingScatterPipeline = device.createComputePipeline({
      label: "spark_wgsl_gpu_counting_scatter_pipeline",
      layout: gpuCountingLayout,
      compute: {
        module: gpuCountingModule,
        entryPoint: "scatter",
      },
    });
    this.renderPipeline = device.createRenderPipeline({
      label: "spark_wgsl_render_pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupLayout],
      }),
      vertex: {
        module: device.createShaderModule({ code: renderWGSL }),
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3",
              },
            ],
          },
        ],
      },
      fragment: {
        module: device.createShaderModule({ code: renderWGSL }),
        entryPoint: "fs_main",
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });
    const quadGeometry = createBatchedQuadGeometry(device);
    this.quadVertexBuffer = quadGeometry.vertexBuffer;
    this.quadIndexBuffer = quadGeometry.indexBuffer;
    this.chunks = createSplatChunks({
      device,
      packedSplats,
      projectBindGroupLayout,
      renderBindGroupLayout,
      uniformByteLength: this.uniformData.byteLength,
    });
  }

  updateOrder(indices: Uint32Array | null, visibleCount?: number) {
    const effectiveVisibleCount = visibleCount ?? this.count;
    for (const chunk of this.chunks) {
      const backSlot = this.frontOrderSlot === 0 ? 1 : 0;
      const order = indices
        ? createChunkOrderData(
            indices,
            effectiveVisibleCount,
            chunk.base,
            chunk.count,
            chunk.textureWidth,
          )
        : { data: chunk.identityOrderData, count: chunk.count };
      chunk.drawCounts[backSlot] = order.count;
      this.device.queue.writeTexture(
        { texture: chunk.orderTextures[backSlot] },
        order.data,
        {
          bytesPerRow: chunk.textureWidth * Uint32Array.BYTES_PER_ELEMENT,
          rowsPerImage: chunk.textureHeight,
        },
        { width: chunk.textureWidth, height: chunk.textureHeight },
      );
    }
    this.frontOrderSlot = this.frontOrderSlot === 0 ? 1 : 0;
  }

  setRenderOptions({
    minAlpha,
    alphaBias,
    alphaRadiusScale,
    blurAmount,
  }: {
    minAlpha?: number;
    alphaBias?: number;
    alphaRadiusScale?: number;
    blurAmount?: number;
  }) {
    this.options = {
      ...this.options,
      minAlpha: minAlpha ?? this.options.minAlpha,
      alphaBias: alphaBias ?? this.options.alphaBias,
      alphaRadiusScale: alphaRadiusScale ?? this.options.alphaRadiusScale,
      blurAmount: blurAmount ?? this.options.blurAmount,
    };
    this.invalidateRenderState("render-options");
  }

  updateOrderGPU(
    positions: Float32Array,
    localToView: Float32Array,
    radial: boolean,
    maxSplats: number,
    algorithm: "bitonic" | "radix" | "bucket" | "counting",
    bucketBits: number,
    bucketCount: number,
    radixEnabled: boolean,
    debugOnce: boolean,
    timestamp: boolean,
  ) {
    if (algorithm === "counting") {
      return this.updateOrderGPUCounting(
        positions,
        localToView,
        radial,
        maxSplats,
        bucketCount,
        debugOnce,
        timestamp,
      );
    }
    if (algorithm === "radix" || algorithm === "bucket") {
      if (!radixEnabled) {
        return {
          ok: false,
          reason: "radix-disabled",
        } satisfies GPUOrderUpdateResult;
      }
      return this.updateOrderGPURadix(
        positions,
        localToView,
        radial,
        maxSplats,
        algorithm === "bucket" ? clampGPUSortBucketBits(bucketBits) : 32,
        debugOnce,
        timestamp,
      );
    }
    const count = this.count;
    if (this.chunks.length !== 1 || count > maxSplats) {
      return {
        ok: false,
        reason: this.chunks.length !== 1 ? "multi-chunk" : "max-splats",
      } satisfies GPUOrderUpdateResult;
    }
    const chunk = this.chunks[0];
    const backSlot = this.frontOrderSlot === 0 ? 1 : 0;
    if (count === 0) {
      chunk.drawCounts[backSlot] = count;
      this.frontOrderSlot = backSlot;
      return {
        ok: true,
        activeCount: 0,
        complete: Promise.resolve(),
      } satisfies GPUOrderUpdateResult;
    }
    const paddedCount = nextPowerOfTwo(count);
    if (paddedCount > maxSplats) {
      return {
        ok: false,
        reason: "padded-max-splats",
      } satisfies GPUOrderUpdateResult;
    }
    const dispatchCount = Math.ceil(paddedCount / gpuSortWorkgroupSize);
    if (dispatchCount > this.device.limits.maxComputeWorkgroupsPerDimension) {
      return {
        ok: false,
        reason: "dispatch-limit",
      } satisfies GPUOrderUpdateResult;
    }
    const needsNewState =
      !chunk.gpuSort || chunk.gpuSort.paddedCount !== paddedCount;
    if (needsNewState) {
      const sortState = this.createGPUSortState(
        positions,
        paddedCount,
        chunk.textureWidth * chunk.textureHeight,
      );
      if (!sortState) {
        return {
          ok: false,
          reason: "resource-limit",
        } satisfies GPUOrderUpdateResult;
      }
      chunk.gpuSort = sortState;
    }
    const sortState = chunk.gpuSort;
    if (!sortState || sortState.paddedCount !== paddedCount) {
      return {
        ok: false,
        reason: "state-mismatch",
      } satisfies GPUOrderUpdateResult;
    }
    this.gpuSortUniformFloatData.set(localToView, 0);
    this.gpuSortUniformUintData[16] = count;
    this.gpuSortUniformUintData[17] = paddedCount;
    this.gpuSortUniformUintData[18] = radial ? 1 : 0;
    this.gpuSortUniformUintData[19] = 0;
    this.gpuSortUniformUintData[20] = 0;
    this.gpuSortUniformUintData[21] = 0;
    this.device.queue.writeBuffer(
      sortState.uniformBuffer,
      0,
      this.gpuSortUniformBufferData,
    );

    const encoder = this.device.createCommandEncoder({
      label: "spark_wgsl_gpu_sort",
    });
    const keyPass = encoder.beginComputePass({
      label: "spark_wgsl_gpu_sort_keys",
    });
    keyPass.setPipeline(this.gpuSortKeyPipeline);
    keyPass.setBindGroup(0, sortState.bindGroup);
    keyPass.dispatchWorkgroups(dispatchCount);
    keyPass.end();

    for (let k = 2; k <= paddedCount; k *= 2) {
      for (let j = k / 2; j >= 1; j /= 2) {
        this.gpuSortUniformUintData[19] = k;
        this.gpuSortUniformUintData[20] = j;
        const passState = this.getGPUSortPassState(sortState, k, j);
        this.device.queue.writeBuffer(
          passState.uniformBuffer,
          0,
          this.gpuSortUniformBufferData,
        );
        const pass = encoder.beginComputePass({
          label: "spark_wgsl_gpu_sort_pass",
        });
        pass.setPipeline(this.gpuSortPassPipeline);
        pass.setBindGroup(0, passState.bindGroup);
        pass.dispatchWorkgroups(dispatchCount);
        pass.end();
      }
    }
    const extractPass = encoder.beginComputePass({
      label: "spark_wgsl_gpu_sort_extract",
    });
    extractPass.setPipeline(this.gpuSortExtractPipeline);
    extractPass.setBindGroup(0, sortState.bindGroup);
    extractPass.dispatchWorkgroups(Math.ceil(count / gpuSortWorkgroupSize));
    extractPass.end();
    encoder.copyBufferToTexture(
      {
        buffer: sortState.orderBuffer,
        bytesPerRow: chunk.textureWidth * Uint32Array.BYTES_PER_ELEMENT,
      },
      { texture: chunk.orderTextures[backSlot] },
      { width: chunk.textureWidth, height: chunk.textureHeight },
    );
    let debugTarget: GPUSortDebugTarget | undefined;
    if (debugOnce) {
      debugTarget = this.createGPUSortDebugTarget(
        encoder,
        sortState.orderBuffer,
        count,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    const complete = this.device.queue.onSubmittedWorkDone();
    const debugReadback = debugTarget
      ? this.mapGPUSortDebugTarget(debugTarget)
      : undefined;
    chunk.drawCounts[backSlot] = count;
    this.frontOrderSlot = backSlot;
    return {
      ok: true,
      activeCount: count,
      complete,
      debugReadback,
    } satisfies GPUOrderUpdateResult;
  }

  private updateOrderGPURadix(
    positions: Float32Array,
    localToView: Float32Array,
    radial: boolean,
    maxSplats: number,
    keyBits: number,
    debugOnce: boolean,
    timestamp: boolean,
  ): GPUOrderUpdateResult {
    const count = this.count;
    if (this.chunks.length !== 1 || count > maxSplats) {
      return {
        ok: false,
        reason: this.chunks.length !== 1 ? "multi-chunk" : "max-splats",
      };
    }
    if (
      this.device.limits.maxComputeWorkgroupStorageSize <
      gpuRadixScatterWorkgroupBytes
    ) {
      return { ok: false, reason: "workgroup-storage-limit" };
    }
    const chunk = this.chunks[0];
    const backSlot = this.frontOrderSlot === 0 ? 1 : 0;
    if (count === 0) {
      chunk.drawCounts[backSlot] = count;
      this.frontOrderSlot = backSlot;
      return {
        ok: true,
        activeCount: 0,
        complete: Promise.resolve(),
      };
    }
    const blockCount = Math.ceil(count / gpuSortWorkgroupSize);
    const blockGroupCount = Math.ceil(
      blockCount / gpuRadixBlockPrefixGroupSize,
    );
    const needsNewState =
      !chunk.gpuRadixSort ||
      chunk.gpuRadixSort.count !== count ||
      chunk.gpuRadixSort.blockCount !== blockCount ||
      chunk.gpuRadixSort.blockGroupCount !== blockGroupCount;
    if (needsNewState) {
      const sortStateResult = this.createGPURadixSortState(
        positions,
        count,
        chunk.textureWidth * chunk.textureHeight,
      );
      if (!sortStateResult.ok) {
        return { ok: false, reason: sortStateResult.reason };
      }
      chunk.gpuRadixSort = sortStateResult.state;
    }
    const sortState = chunk.gpuRadixSort;
    if (!sortState || sortState.count !== count) {
      return { ok: false, reason: "state-mismatch" };
    }
    const dispatchCount = sortState.blockCount;
    this.gpuRadixUniformFloatData.set(localToView, 0);
    this.gpuRadixUniformUintData[16] = count;
    this.gpuRadixUniformUintData[17] = 0;
    this.gpuRadixUniformUintData[18] = radial ? 1 : 0;
    this.gpuRadixUniformUintData[20] = sortState.blockCount;
    this.gpuRadixUniformUintData[21] = sortState.blockGroupCount;
    this.gpuRadixUniformUintData[22] = keyBits;

    const encoder = this.device.createCommandEncoder({
      label: "spark_wgsl_gpu_radix_sort",
    });
    const timestampState =
      timestamp && !this.timestampReadPending
        ? this.resetTimestampState()
        : null;
    this.gpuRadixUniformUintData[17] = 0;
    this.gpuRadixUniformUintData[19] = 0;
    const keyState = this.getGPURadixPassState(sortState, "keys");
    this.device.queue.writeBuffer(
      keyState.uniformBuffer,
      0,
      this.gpuRadixUniformBufferData,
    );
    const keyPass = encoder.beginComputePass(
      this.timestampPassDescriptor(timestampState, "spark_wgsl_gpu_radix_keys"),
    );
    keyPass.setPipeline(this.gpuRadixKeyPipeline);
    keyPass.setBindGroup(0, keyState.bindGroup);
    keyPass.dispatchWorkgroups(dispatchCount);
    keyPass.end();

    const passCount = Math.ceil(keyBits / 8);
    for (let passIndex = 0; passIndex < passCount; passIndex++) {
      const shift = passIndex * 8;
      this.gpuRadixUniformUintData[17] = shift;
      this.gpuRadixUniformUintData[19] = passIndex;
      const passState = this.getGPURadixPassState(
        sortState,
        `pass:${passIndex}`,
      );
      this.device.queue.writeBuffer(
        passState.uniformBuffer,
        0,
        this.gpuRadixUniformBufferData,
      );
      const histogramPass = encoder.beginComputePass(
        this.timestampPassDescriptor(
          timestampState,
          "spark_wgsl_gpu_radix_histogram",
        ),
      );
      histogramPass.setPipeline(this.gpuRadixHistogramPipeline);
      histogramPass.setBindGroup(0, passState.bindGroup);
      histogramPass.dispatchWorkgroups(dispatchCount);
      histogramPass.end();

      const bucketTotalPass = encoder.beginComputePass(
        this.timestampPassDescriptor(
          timestampState,
          "spark_wgsl_gpu_radix_bucket_totals",
        ),
      );
      bucketTotalPass.setPipeline(this.gpuRadixBucketTotalPipeline);
      bucketTotalPass.setBindGroup(0, passState.bindGroup);
      bucketTotalPass.dispatchWorkgroups(1, sortState.blockGroupCount);
      bucketTotalPass.end();

      const bucketBasePass = encoder.beginComputePass(
        this.timestampPassDescriptor(
          timestampState,
          "spark_wgsl_gpu_radix_bucket_bases",
        ),
      );
      bucketBasePass.setPipeline(this.gpuRadixBucketBasePipeline);
      bucketBasePass.setBindGroup(0, passState.bindGroup);
      bucketBasePass.dispatchWorkgroups(1);
      bucketBasePass.end();

      const blockPrefixPass = encoder.beginComputePass(
        this.timestampPassDescriptor(
          timestampState,
          "spark_wgsl_gpu_radix_block_prefix",
        ),
      );
      blockPrefixPass.setPipeline(this.gpuRadixBlockPrefixPipeline);
      blockPrefixPass.setBindGroup(0, passState.bindGroup);
      blockPrefixPass.dispatchWorkgroups(1, sortState.blockGroupCount);
      blockPrefixPass.end();

      const scatterPass = encoder.beginComputePass(
        this.timestampPassDescriptor(
          timestampState,
          "spark_wgsl_gpu_radix_scatter",
        ),
      );
      scatterPass.setPipeline(this.gpuRadixScatterPipeline);
      scatterPass.setBindGroup(0, passState.bindGroup);
      scatterPass.dispatchWorkgroups(dispatchCount);
      scatterPass.end();
    }

    this.gpuRadixUniformUintData[19] = passCount;
    const extractState = this.getGPURadixPassState(sortState, "extract");
    this.device.queue.writeBuffer(
      extractState.uniformBuffer,
      0,
      this.gpuRadixUniformBufferData,
    );
    const extractPass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_extract",
      ),
    );
    extractPass.setPipeline(this.gpuRadixExtractPipeline);
    extractPass.setBindGroup(0, extractState.bindGroup);
    extractPass.dispatchWorkgroups(dispatchCount);
    extractPass.end();
    encoder.copyBufferToTexture(
      {
        buffer: sortState.orderBuffer,
        bytesPerRow: chunk.textureWidth * Uint32Array.BYTES_PER_ELEMENT,
      },
      { texture: chunk.orderTextures[backSlot] },
      { width: chunk.textureWidth, height: chunk.textureHeight },
    );
    let debugTarget: GPUSortDebugTarget | undefined;
    if (debugOnce) {
      debugTarget = this.createGPUSortDebugTarget(
        encoder,
        sortState.orderBuffer,
        count,
      );
    }
    this.resolveTimestampState(encoder, timestampState);
    this.device.queue.submit([encoder.finish()]);
    if (timestampState) {
      this.timestampReadPending = true;
      void this.readTimestampState(timestampState)
        .then((timings) => {
          if (timings) {
            this.lastGPURadixTimings = timings;
            console.log("Spark WebGPU radix timings", timings);
          }
        })
        .finally(() => {
          this.timestampReadPending = false;
        });
    }
    const complete = this.device.queue.onSubmittedWorkDone();
    const debugReadback = debugTarget
      ? this.mapGPUSortDebugTarget(debugTarget)
      : undefined;
    chunk.drawCounts[backSlot] = count;
    this.frontOrderSlot = backSlot;
    return { ok: true, activeCount: count, complete, debugReadback };
  }

  private updateOrderGPUCounting(
    positions: Float32Array,
    localToView: Float32Array,
    radial: boolean,
    maxSplats: number,
    requestedBucketCount: number,
    debugOnce: boolean,
    timestamp: boolean,
  ): GPUOrderUpdateResult {
    const count = this.count;
    if (this.chunks.length !== 1 || count > maxSplats) {
      return {
        ok: false,
        reason: this.chunks.length !== 1 ? "multi-chunk" : "max-splats",
      };
    }
    const bucketCount = clampGPUCountingBucketCount(requestedBucketCount);
    const chunk = this.chunks[0];
    const backSlot = this.frontOrderSlot === 0 ? 1 : 0;
    if (count === 0) {
      chunk.drawCounts[backSlot] = count;
      this.frontOrderSlot = backSlot;
      return {
        ok: true,
        activeCount: 0,
        complete: Promise.resolve(),
      };
    }
    const dispatchCount = Math.ceil(count / gpuSortWorkgroupSize);
    if (dispatchCount > this.device.limits.maxComputeWorkgroupsPerDimension) {
      return { ok: false, reason: "dispatch-limit" };
    }
    const needsNewState =
      !chunk.gpuCountingSort ||
      chunk.gpuCountingSort.count !== count ||
      chunk.gpuCountingSort.bucketCount !== bucketCount;
    if (needsNewState) {
      const sortStateResult = this.createGPUCountingSortState(
        positions,
        count,
        chunk.textureWidth * chunk.textureHeight,
        bucketCount,
      );
      if (!sortStateResult.ok) {
        return { ok: false, reason: sortStateResult.reason };
      }
      chunk.gpuCountingSort = sortStateResult.state;
    }
    const sortState = chunk.gpuCountingSort;
    if (
      !sortState ||
      sortState.count !== count ||
      sortState.bucketCount !== bucketCount
    ) {
      return { ok: false, reason: "state-mismatch" };
    }

    this.gpuCountingUniformFloatData.set(localToView, 0);
    this.gpuCountingUniformUintData[16] = count;
    this.gpuCountingUniformUintData[17] = bucketCount;
    this.gpuCountingUniformUintData[18] = radial ? 1 : 0;
    this.gpuCountingUniformUintData[19] = 32 - Math.log2(bucketCount);
    this.device.queue.writeBuffer(
      sortState.uniformBuffer,
      0,
      this.gpuCountingUniformBufferData,
    );

    const encoder = this.device.createCommandEncoder({
      label: "spark_wgsl_gpu_counting_sort",
    });
    const timestampState =
      timestamp && !this.timestampReadPending
        ? this.resetTimestampState()
        : null;
    encoder.clearBuffer(
      sortState.histogramBuffer,
      0,
      bucketCount * Uint32Array.BYTES_PER_ELEMENT,
    );
    const keyHistogramPass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_histogram",
      ),
    );
    keyHistogramPass.setPipeline(this.gpuCountingKeyHistogramPipeline);
    keyHistogramPass.setBindGroup(0, sortState.bindGroup);
    keyHistogramPass.dispatchWorkgroups(dispatchCount);
    keyHistogramPass.end();

    const blockPrefixPass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_bucket_bases",
      ),
    );
    blockPrefixPass.setPipeline(this.gpuCountingBlockPrefixPipeline);
    blockPrefixPass.setBindGroup(0, sortState.bindGroup);
    blockPrefixPass.dispatchWorkgroups(bucketCount / gpuCountingScanBlockSize);
    blockPrefixPass.end();

    const groupPrefixPass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_bucket_bases",
      ),
    );
    groupPrefixPass.setPipeline(this.gpuCountingGroupPrefixPipeline);
    groupPrefixPass.setBindGroup(0, sortState.bindGroup);
    groupPrefixPass.dispatchWorkgroups(1);
    groupPrefixPass.end();

    const addGroupBasePass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_bucket_bases",
      ),
    );
    addGroupBasePass.setPipeline(this.gpuCountingAddGroupBasePipeline);
    addGroupBasePass.setBindGroup(0, sortState.bindGroup);
    addGroupBasePass.dispatchWorkgroups(bucketCount / gpuCountingScanBlockSize);
    addGroupBasePass.end();

    const scatterPass = encoder.beginComputePass(
      this.timestampPassDescriptor(
        timestampState,
        "spark_wgsl_gpu_radix_scatter",
      ),
    );
    scatterPass.setPipeline(this.gpuCountingScatterPipeline);
    scatterPass.setBindGroup(0, sortState.bindGroup);
    scatterPass.dispatchWorkgroups(dispatchCount);
    scatterPass.end();

    encoder.copyBufferToTexture(
      {
        buffer: sortState.orderBuffer,
        bytesPerRow: chunk.textureWidth * Uint32Array.BYTES_PER_ELEMENT,
      },
      { texture: chunk.orderTextures[backSlot] },
      { width: chunk.textureWidth, height: chunk.textureHeight },
    );
    let debugTarget: GPUSortDebugTarget | undefined;
    if (debugOnce) {
      debugTarget = this.createGPUSortDebugTarget(
        encoder,
        sortState.orderBuffer,
        count,
      );
    }
    this.resolveTimestampState(encoder, timestampState);
    this.device.queue.submit([encoder.finish()]);
    if (timestampState) {
      this.timestampReadPending = true;
      void this.readTimestampState(timestampState)
        .then((timings) => {
          if (timings) {
            this.lastGPURadixTimings = timings;
            console.log("Spark WebGPU counting timings", timings);
          }
        })
        .finally(() => {
          this.timestampReadPending = false;
        });
    }
    const complete = this.device.queue.onSubmittedWorkDone();
    const debugReadback = debugTarget
      ? this.mapGPUSortDebugTarget(debugTarget)
      : undefined;
    chunk.drawCounts[backSlot] = count;
    this.frontOrderSlot = backSlot;
    return { ok: true, activeCount: count, complete, debugReadback };
  }

  private createGPUSortState(
    positions: Float32Array,
    paddedCount: number,
    orderCapacity: number,
  ): GPUSortState | null {
    const positionsBuffer = createBuffer(
      this.device,
      "spark_wgsl_gpu_sort_positions",
      positions,
      GPUBufferUsage.STORAGE,
    );
    const pairsBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_sort_pairs",
      size: paddedCount * 2 * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const orderBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_sort_order",
      size: orderCapacity * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const uniformBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_sort_uniforms",
      size: this.gpuSortUniformBufferData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuSortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer } },
        { binding: 1, resource: { buffer: pairsBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
        { binding: 3, resource: { buffer: orderBuffer } },
      ],
    });
    return {
      paddedCount,
      positionsBuffer,
      pairsBuffer,
      orderBuffer,
      uniformBuffer,
      bindGroup,
      passStates: new Map(),
    };
  }

  private getGPUSortPassState(sortState: GPUSortState, k: number, j: number) {
    const key = `${k}:${j}`;
    const cached = sortState.passStates.get(key);
    if (cached) {
      return cached;
    }
    const uniformBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_sort_pass_uniforms",
      size: this.gpuSortUniformBufferData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuSortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: sortState.positionsBuffer } },
        { binding: 1, resource: { buffer: sortState.pairsBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
        { binding: 3, resource: { buffer: sortState.orderBuffer } },
      ],
    });
    const passState = { uniformBuffer, bindGroup };
    sortState.passStates.set(key, passState);
    return passState;
  }

  private createGPUSortDebugTarget(
    encoder: GPUCommandEncoder,
    orderBuffer: GPUBuffer,
    count: number,
  ): GPUSortDebugTarget {
    const byteLength = count * Uint32Array.BYTES_PER_ELEMENT;
    const readBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_sort_debug_readback",
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(orderBuffer, 0, readBuffer, 0, byteLength);
    return { readBuffer };
  }

  private mapGPUSortDebugTarget(
    target: GPUSortDebugTarget,
  ): GPUSortDebugReadback {
    return {
      promise: target.readBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const order = new Uint32Array(
          target.readBuffer.getMappedRange().slice(0),
        );
        target.readBuffer.unmap();
        return order;
      }),
    };
  }

  private createTimestampState(): GPUTimestampState | null {
    if (!this.supportsTimestampQuery) {
      return null;
    }
    const queryCount = 64;
    const byteLength = queryCount * BigUint64Array.BYTES_PER_ELEMENT;
    const querySet = this.device.createQuerySet({
      type: "timestamp",
      count: queryCount,
    });
    const resolveBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_timestamp_resolve",
      size: byteLength,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_timestamp_readback",
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    return { querySet, resolveBuffer, readBuffer, nextQuery: 0, labels: [] };
  }

  private resetTimestampState(): GPUTimestampState | null {
    this.timestampState ??= this.createTimestampState();
    if (!this.timestampState) {
      return null;
    }
    this.timestampState.nextQuery = 0;
    this.timestampState.labels = [];
    return this.timestampState;
  }

  private timestampPassDescriptor(
    state: GPUTimestampState | null,
    label: string,
  ): GPUComputePassDescriptor {
    return this.timestampDescriptor(state, label);
  }

  private timestampRenderPassDescriptor(
    state: GPUTimestampState | null,
    label: string,
    descriptor: GPURenderPassDescriptor,
  ): GPURenderPassDescriptor {
    return { ...descriptor, ...this.timestampDescriptor(state, label) };
  }

  private timestampDescriptor(
    state: GPUTimestampState | null,
    label: string,
  ): {
    label: string;
    timestampWrites?: {
      querySet: GPUQuerySet;
      beginningOfPassWriteIndex: number;
      endOfPassWriteIndex: number;
    };
  } {
    if (!state || state.nextQuery + 2 > 64) {
      return { label };
    }
    const beginningOfPassWriteIndex = state.nextQuery++;
    const endOfPassWriteIndex = state.nextQuery++;
    state.labels.push(label);
    return {
      label,
      timestampWrites: {
        querySet: state.querySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex,
      },
    };
  }

  private resolveTimestampState(
    encoder: GPUCommandEncoder,
    state: GPUTimestampState | null,
  ) {
    if (!state || state.nextQuery === 0) {
      return;
    }
    const byteLength = state.nextQuery * BigUint64Array.BYTES_PER_ELEMENT;
    encoder.resolveQuerySet(
      state.querySet,
      0,
      state.nextQuery,
      state.resolveBuffer,
      0,
    );
    encoder.copyBufferToBuffer(
      state.resolveBuffer,
      0,
      state.readBuffer,
      0,
      byteLength,
    );
  }

  private readTimestampState(
    state: GPUTimestampState | null,
  ): Promise<GPURadixTimingResult | null> {
    if (!state || state.nextQuery === 0) {
      return Promise.resolve(null);
    }
    const queryCount = state.nextQuery;
    const labels = state.labels.slice();
    return state.readBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const values = new BigUint64Array(
          state.readBuffer.getMappedRange().slice(0),
        );
        state.readBuffer.unmap();
        const timings: Record<string, number> = {};
        for (let i = 0; i < labels.length; i++) {
          const start = values[i * 2];
          const end = values[i * 2 + 1];
          const durationNs = end > start ? Number(end - start) : 0;
          timings[labels[i]] =
            (timings[labels[i]] ?? 0) + durationNs / 1_000_000;
        }
        const totalMs = Object.values(timings).reduce(
          (sum, value) => sum + value,
          0,
        );
        return {
          totalMs,
          keysMs: timings.spark_wgsl_gpu_radix_keys ?? 0,
          histogramMs: timings.spark_wgsl_gpu_radix_histogram ?? 0,
          bucketTotalsMs: timings.spark_wgsl_gpu_radix_bucket_totals ?? 0,
          bucketBasesMs: timings.spark_wgsl_gpu_radix_bucket_bases ?? 0,
          blockPrefixMs: timings.spark_wgsl_gpu_radix_block_prefix ?? 0,
          scatterMs: timings.spark_wgsl_gpu_radix_scatter ?? 0,
          extractMs: timings.spark_wgsl_gpu_radix_extract ?? 0,
          copyMs: 0,
          projectMs: timings.spark_wgsl_project ?? 0,
          renderMs: timings.spark_wgsl_render ?? 0,
        };
      })
      .catch((error) => {
        console.warn("Spark WebGPU timestamp readback failed", error);
        return null;
      })
      .finally(() => {
        void queryCount;
      });
  }

  private createGPURadixSortState(
    positions: Float32Array,
    count: number,
    orderCapacity: number,
  ): GPURadixSortStateResult {
    const blockCount = Math.ceil(count / gpuSortWorkgroupSize);
    const blockGroupCount = Math.ceil(
      blockCount / gpuRadixBlockPrefixGroupSize,
    );
    if (blockCount > this.device.limits.maxComputeWorkgroupsPerDimension) {
      return { ok: false, reason: "block-dispatch-limit" };
    }
    if (blockGroupCount > this.device.limits.maxComputeWorkgroupsPerDimension) {
      return { ok: false, reason: "block-group-dispatch-limit" };
    }
    const pairByteLength = count * 2 * Uint32Array.BYTES_PER_ELEMENT;
    const orderByteLength = orderCapacity * Uint32Array.BYTES_PER_ELEMENT;
    const histogramByteLength =
      blockCount * 256 * Uint32Array.BYTES_PER_ELEMENT;
    const bucketPrefixByteLength =
      (512 + blockGroupCount * 256) * Uint32Array.BYTES_PER_ELEMENT;
    if (!this.fitsStorageBuffer(positions.byteLength)) {
      return { ok: false, reason: "positions-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(pairByteLength)) {
      return { ok: false, reason: "pair-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(orderByteLength)) {
      return { ok: false, reason: "order-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(histogramByteLength)) {
      return { ok: false, reason: "histogram-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(bucketPrefixByteLength)) {
      return { ok: false, reason: "bucket-prefix-buffer-limit" };
    }
    const positionsBuffer = createBuffer(
      this.device,
      "spark_wgsl_gpu_radix_positions",
      positions,
      GPUBufferUsage.STORAGE,
    );
    const pairsA = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_pairs_a",
      size: pairByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const pairsB = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_pairs_b",
      size: pairByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const orderBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_order",
      size: orderByteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const histogramBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_histogram",
      size: histogramByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const prefixBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_prefix",
      size: histogramByteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    const bucketPrefixBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_bucket_prefix",
      size: bucketPrefixByteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    const uniformBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_uniforms",
      size: this.gpuRadixUniformBufferData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuRadixBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer } },
        { binding: 1, resource: { buffer: pairsA } },
        { binding: 2, resource: { buffer: pairsB } },
        { binding: 3, resource: { buffer: orderBuffer } },
        { binding: 4, resource: { buffer: histogramBuffer } },
        { binding: 5, resource: { buffer: prefixBuffer } },
        { binding: 6, resource: { buffer: bucketPrefixBuffer } },
        { binding: 7, resource: { buffer: uniformBuffer } },
      ],
    });
    return {
      ok: true,
      state: {
        count,
        blockCount,
        blockGroupCount,
        positionsBuffer,
        pairsA,
        pairsB,
        orderBuffer,
        histogramBuffer,
        bucketPrefixBuffer,
        prefixBuffer,
        uniformBuffer,
        bindGroup,
        passStates: new Map(),
      },
    };
  }

  private createGPUCountingSortState(
    positions: Float32Array,
    count: number,
    orderCapacity: number,
    bucketCount: number,
  ): GPUCountingSortStateResult {
    const keyByteLength = count * Uint32Array.BYTES_PER_ELEMENT;
    const orderByteLength = orderCapacity * Uint32Array.BYTES_PER_ELEMENT;
    const bucketByteLength = bucketCount * Uint32Array.BYTES_PER_ELEMENT;
    const groupCount = bucketCount / gpuCountingScanBlockSize;
    const groupTotalsByteLength = groupCount * Uint32Array.BYTES_PER_ELEMENT;
    if (!this.fitsStorageBuffer(positions.byteLength)) {
      return { ok: false, reason: "positions-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(keyByteLength)) {
      return { ok: false, reason: "key-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(orderByteLength)) {
      return { ok: false, reason: "order-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(bucketByteLength)) {
      return { ok: false, reason: "bucket-buffer-limit" };
    }
    if (!this.fitsStorageBuffer(groupTotalsByteLength)) {
      return { ok: false, reason: "group-total-buffer-limit" };
    }
    const positionsBuffer = createBuffer(
      this.device,
      "spark_wgsl_gpu_counting_positions",
      positions,
      GPUBufferUsage.STORAGE,
    );
    const keysBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_keys",
      size: keyByteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    const orderBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_order",
      size: orderByteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const histogramBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_histogram",
      size: bucketByteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    const offsetsBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_offsets",
      size: bucketByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const groupTotalsBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_group_totals",
      size: groupTotalsByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const uniformBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_counting_uniforms",
      size: this.gpuCountingUniformBufferData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuCountingBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsBuffer } },
        { binding: 1, resource: { buffer: keysBuffer } },
        { binding: 2, resource: { buffer: orderBuffer } },
        { binding: 3, resource: { buffer: histogramBuffer } },
        { binding: 4, resource: { buffer: offsetsBuffer } },
        { binding: 5, resource: { buffer: groupTotalsBuffer } },
        { binding: 6, resource: { buffer: uniformBuffer } },
      ],
    });
    return {
      ok: true,
      state: {
        count,
        bucketCount,
        positionsBuffer,
        keysBuffer,
        orderBuffer,
        histogramBuffer,
        offsetsBuffer,
        groupTotalsBuffer,
        uniformBuffer,
        bindGroup,
      },
    };
  }

  private getGPURadixPassState(sortState: GPURadixSortState, key: string) {
    const cached = sortState.passStates.get(key);
    if (cached) {
      return cached;
    }
    const uniformBuffer = this.device.createBuffer({
      label: "spark_wgsl_gpu_radix_pass_uniforms",
      size: this.gpuRadixUniformBufferData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.gpuRadixBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: sortState.positionsBuffer } },
        { binding: 1, resource: { buffer: sortState.pairsA } },
        { binding: 2, resource: { buffer: sortState.pairsB } },
        { binding: 3, resource: { buffer: sortState.orderBuffer } },
        { binding: 4, resource: { buffer: sortState.histogramBuffer } },
        { binding: 5, resource: { buffer: sortState.prefixBuffer } },
        { binding: 6, resource: { buffer: sortState.bucketPrefixBuffer } },
        { binding: 7, resource: { buffer: uniformBuffer } },
      ],
    });
    const passState = { uniformBuffer, bindGroup };
    sortState.passStates.set(key, passState);
    return passState;
  }

  private fitsStorageBuffer(byteLength: number) {
    const maxBindingSize = this.device.limits.maxStorageBufferBindingSize;
    return (
      Number.isFinite(byteLength) &&
      byteLength > 0 &&
      byteLength <= maxBindingSize
    );
  }

  render(
    renderer: WebGPURendererLike,
    camera: THREE.Camera,
    object: THREE.Object3D,
    clear: boolean,
    { force = false }: { force?: boolean } = {},
  ) {
    renderer.getDrawingBufferSize?.(this.drawingBufferSize);
    camera.getWorldPosition(this.cameraPosition);
    this.localCameraPosition
      .copy(this.cameraPosition)
      .applyMatrix4(scratchWorldInverse.copy(object.matrixWorld).invert());
    camera.getWorldDirection(this.cameraDirection);
    this.writeUniformState(camera, object, false);
    this.updateProjectionState();

    if (!force && !this.isRenderStateDirty()) {
      this.skippedStableCount++;
      return;
    }
    if (this.inFlightFrameCount >= maxInFlightFrames) {
      this.deferredRenderRequested = true;
      this.skippedBusyCount++;
      this.skippedBusyRenderCount++;
      this.lastBusyPass = "render";
      this.lastBusyDirtyReason = this.lastDirtyReason;
      return;
    }
    const encoder = this.device.createCommandEncoder({
      label: "spark_wgsl_frame",
    });
    const timestampState =
      this.options.gpuSortTimestamp &&
      this.supportsTimestampQuery &&
      !this.timestampReadPending
        ? this.resetTimestampState()
        : null;
    const orderSlot = this.frontOrderSlot;
    for (const chunk of this.chunks) {
      this.uniformData[34] = chunk.drawCounts[orderSlot];
      this.uniformData[40] = chunk.textureWidth;
      this.uniformData[41] = chunk.base;
      this.uniformData[42] = this.options.highPrecisionProjected ? 1 : 0;
      this.device.queue.writeBuffer(chunk.uniformBuffer, 0, this.uniformData);
    }

    const projectPass = encoder.beginComputePass(
      this.timestampPassDescriptor(timestampState, "spark_wgsl_project"),
    );
    projectPass.setPipeline(this.projectPipeline);
    let projectDispatches = 0;
    for (const chunk of this.chunks) {
      if (chunk.drawCounts[orderSlot] === 0) {
        continue;
      }
      projectPass.setBindGroup(0, chunk.projectBindGroups[orderSlot]);
      projectPass.dispatchWorkgroups(
        Math.ceil(chunk.drawCounts[orderSlot] / projectWorkgroupSize),
      );
      projectDispatches++;
    }
    projectPass.end();

    renderer.getClearColor?.(this.clearColor);
    const renderPass = encoder.beginRenderPass(
      this.timestampRenderPassDescriptor(timestampState, "spark_wgsl_render", {
        colorAttachments: [
          {
            view: this.context.getCurrentTexture().createView(),
            clearValue: {
              r: this.clearColor.r,
              g: this.clearColor.g,
              b: this.clearColor.b,
              a: renderer.getClearAlpha?.() ?? 1,
            },
            loadOp: clear ? "clear" : "load",
            storeOp: "store",
          },
        ],
      }),
    );
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setVertexBuffer(0, this.quadVertexBuffer);
    renderPass.setIndexBuffer(this.quadIndexBuffer, "uint32");
    let drawCalls = 0;
    let drawSplats = 0;
    for (const chunk of this.chunks) {
      if (chunk.drawCounts[orderSlot] === 0) {
        continue;
      }
      renderPass.setBindGroup(0, chunk.renderBindGroup);
      renderPass.drawIndexed(
        batchedQuadIndexCount,
        Math.ceil(chunk.drawCounts[orderSlot] / batchedQuadSplatCount),
      );
      drawCalls++;
      drawSplats += chunk.drawCounts[orderSlot];
    }
    renderPass.end();
    this.lastSubmittedPass = "render";
    this.lastSubmittedProjectDispatches = projectDispatches;
    this.lastSubmittedDrawCalls = drawCalls;
    this.lastSubmittedDrawSplats = drawSplats;
    this.resolveTimestampState(encoder, timestampState);
    this.submitFrame(encoder);
    if (timestampState) {
      this.timestampReadPending = true;
      void this.readTimestampState(timestampState)
        .then((timings) => {
          if (timings) {
            this.lastGPURenderTimings = {
              totalMs: timings.totalMs,
              projectMs: timings.projectMs ?? 0,
              renderMs: timings.renderMs ?? 0,
            };
            console.log(
              "Spark WebGPU render timings",
              this.lastGPURenderTimings,
            );
          }
        })
        .finally(() => {
          this.timestampReadPending = false;
        });
    }
  }

  clear(renderer: WebGPURendererLike) {
    if (this.inFlightFrameCount >= maxInFlightFrames) {
      this.deferredRenderRequested = true;
      this.skippedBusyCount++;
      this.skippedBusyClearCount++;
      this.lastBusyPass = "clear";
      this.lastBusyDirtyReason = "clear";
      return;
    }
    renderer.getClearColor?.(this.clearColor);
    const encoder = this.device.createCommandEncoder({
      label: "spark_wgsl_clear",
    });
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: {
            r: this.clearColor.r,
            g: this.clearColor.g,
            b: this.clearColor.b,
            a: renderer.getClearAlpha?.() ?? 1,
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.end();
    this.clearFrameCount++;
    this.lastSubmittedPass = "clear";
    this.lastSubmittedProjectDispatches = 0;
    this.lastSubmittedDrawCalls = 0;
    this.lastSubmittedDrawSplats = 0;
    this.submitFrame(encoder);
  }

  private isRenderStateDirty() {
    if (!this.hasSubmittedFrame) {
      if (
        this.lastDirtyReason === "stable" ||
        this.lastDirtyReason === "none"
      ) {
        this.lastDirtyReason = "initial";
      }
      this.lastMatrixDelta = Number.POSITIVE_INFINITY;
      this.lastPositionDelta = Number.POSITIVE_INFINITY;
      this.lastDirectionAngle = Number.POSITIVE_INFINITY;
      this.lastProjectionDelta = Number.POSITIVE_INFINITY;
      this.lastScalarDelta = Number.POSITIVE_INFINITY;
      this.lastDirtyIndex = -1;
      return true;
    }
    const positionDelta = this.cameraPosition.distanceTo(
      this.lastCameraPosition,
    );
    const directionDot = THREE.MathUtils.clamp(
      this.cameraDirection.dot(this.lastCameraDirection),
      -1,
      1,
    );
    const directionAngle = Math.acos(directionDot);
    let maxProjectionDelta = 0;
    let maxProjectionIndex = -1;
    let maxScalarDelta = 0;
    let maxScalarIndex = -1;
    for (let i = 0; i < this.projectionState.length; i++) {
      const delta = Math.abs(
        this.projectionState[i] - this.lastProjectionState[i],
      );
      if (delta > maxProjectionDelta) {
        maxProjectionDelta = delta;
        maxProjectionIndex = i;
      }
    }
    for (let i = 32; i < this.uniformData.length; i++) {
      if (i === 34 || i === 40 || i === 41) {
        continue;
      }
      const delta = Math.abs(this.uniformData[i] - this.lastRenderState[i]);
      if (delta > maxScalarDelta) {
        maxScalarDelta = delta;
        maxScalarIndex = i;
      }
    }
    this.lastMatrixDelta = Math.max(
      positionDelta,
      directionAngle,
      maxProjectionDelta,
    );
    this.lastPositionDelta = positionDelta;
    this.lastDirectionAngle = directionAngle;
    this.lastProjectionDelta = maxProjectionDelta;
    this.lastScalarDelta = maxScalarDelta;
    if (positionDelta > this.options.renderEpsilon) {
      this.lastDirtyReason = "position";
      this.lastDirtyIndex = -1;
      return true;
    }
    if (directionAngle > this.options.renderEpsilon) {
      this.lastDirtyReason = "direction";
      this.lastDirtyIndex = -1;
      return true;
    }
    if (maxProjectionDelta > renderStateScalarEpsilon) {
      this.lastDirtyReason = "projection";
      this.lastDirtyIndex = maxProjectionIndex;
      return true;
    }
    if (maxScalarDelta > renderStateScalarEpsilon) {
      this.lastDirtyReason = "scalar";
      this.lastDirtyIndex = maxScalarIndex;
      return true;
    }
    this.lastDirtyReason = "stable";
    this.lastDirtyIndex = -1;
    return false;
  }

  private storeRenderState() {
    this.lastRenderState.set(this.uniformData);
    this.lastProjectionState.set(this.projectionState);
    this.lastCameraPosition.copy(this.cameraPosition);
    this.lastCameraDirection.copy(this.cameraDirection);
    this.hasSubmittedFrame = true;
  }

  invalidateRenderState(reason = "external") {
    this.hasSubmittedFrame = false;
    this.lastDirtyReason = reason;
  }

  requestDeferredRender() {
    this.deferredRenderRequested = true;
  }

  private submitFrame(encoder: GPUCommandEncoder) {
    this.inFlightFrameCount++;
    this.device.queue.submit([encoder.finish()]);
    this.submittedFrameCount++;
    this.device.queue.onSubmittedWorkDone().finally(() => {
      this.inFlightFrameCount = Math.max(0, this.inFlightFrameCount - 1);
      if (
        this.inFlightFrameCount < maxInFlightFrames &&
        this.deferredRenderRequested
      ) {
        this.deferredRenderRequested = false;
        this.onReadyForDeferredRender?.();
      }
    });
    this.storeRenderState();
  }

  private writeUniformState(
    camera: THREE.Camera,
    object: THREE.Object3D,
    sortRadial: boolean,
  ) {
    this.localToView
      .copy(camera.matrixWorldInverse)
      .multiply(object.matrixWorld);
    this.uniformData.set(this.localToView.elements, 0);
    this.uniformData.set(camera.projectionMatrix.elements, 16);
    this.uniformData[32] = this.drawingBufferSize.x;
    this.uniformData[33] = this.drawingBufferSize.y;
    this.uniformData[35] = this.options.sizeScale;
    this.uniformData[36] = this.options.opacityScale;
    this.uniformData[37] = this.options.minAlpha;
    this.uniformData[38] = this.options.maxPixelRadius;
    this.uniformData[39] = this.options.maxStdDev;
    this.uniformData[44] = this.options.blurAmount;
    this.uniformData[45] = this.options.preBlurAmount;
    this.uniformData[46] = this.options.falloff;
    this.uniformData[47] = this.options.alphaBias;
    this.uniformData[51] = this.options.alphaRadiusScale;
    this.uniformData[48] = this.localCameraPosition.x;
    this.uniformData[49] = this.localCameraPosition.y;
    this.uniformData[50] = this.localCameraPosition.z;
    this.uniformData[52] = Math.min(
      this.packedSplats.getNumSh?.() ?? 0,
      this.packedSplats.maxSh ?? 3,
    );
    this.uniformData[53] = this.packedSplats.splatEncoding?.sh1Max ?? 1;
    this.uniformData[54] = this.packedSplats.splatEncoding?.sh2Max ?? 1;
    this.uniformData[55] = this.packedSplats.splatEncoding?.sh3Max ?? 1;
  }

  getDrawCount() {
    return this.chunks.reduce(
      (sum, chunk) => sum + chunk.drawCounts[this.frontOrderSlot],
      0,
    );
  }

  getDebugState() {
    const firstChunk = this.chunks[0];
    const gpuSortBlockCount =
      firstChunk?.gpuRadixSort?.blockCount ??
      (firstChunk?.gpuCountingSort
        ? Math.ceil(firstChunk.gpuCountingSort.count / gpuSortWorkgroupSize)
        : undefined) ??
      (firstChunk?.gpuSort
        ? Math.ceil(firstChunk.gpuSort.paddedCount / gpuSortWorkgroupSize)
        : 0);
    const gpuSortBlockGroupCount =
      firstChunk?.gpuRadixSort?.blockGroupCount ??
      firstChunk?.gpuCountingSort?.bucketCount ??
      0;
    return {
      gpuBusy: this.inFlightFrameCount > 0,
      inFlightFrameCount: this.inFlightFrameCount,
      maxInFlightFrames,
      submittedFrameCount: this.submittedFrameCount,
      skippedStableCount: this.skippedStableCount,
      skippedBusyCount: this.skippedBusyCount,
      skippedBusyRenderCount: this.skippedBusyRenderCount,
      skippedBusyClearCount: this.skippedBusyClearCount,
      clearFrameCount: this.clearFrameCount,
      deferredRenderRequested: this.deferredRenderRequested,
      lastSubmittedPass: this.lastSubmittedPass,
      lastSubmittedProjectDispatches: this.lastSubmittedProjectDispatches,
      lastSubmittedDrawCalls: this.lastSubmittedDrawCalls,
      lastSubmittedDrawSplats: this.lastSubmittedDrawSplats,
      lastBusyPass: this.lastBusyPass,
      lastBusyDirtyReason: this.lastBusyDirtyReason,
      lastDirtyReason: this.lastDirtyReason,
      lastMatrixDelta: this.lastMatrixDelta,
      lastPositionDelta: this.lastPositionDelta,
      lastDirectionAngle: this.lastDirectionAngle,
      lastProjectionDelta: this.lastProjectionDelta,
      lastScalarDelta: this.lastScalarDelta,
      drawCount: this.getDrawCount(),
      chunkCount: this.chunks.length,
      gpuSortBlockCount,
      gpuSortBlockGroupCount,
      lastGPURenderTimings: this.lastGPURenderTimings,
    };
  }

  private updateProjectionState() {
    this.projectionState.set(this.uniformData.subarray(16, 34), 0);
  }
}

function createSplatChunks({
  device,
  packedSplats,
  projectBindGroupLayout,
  renderBindGroupLayout,
  uniformByteLength,
}: {
  device: GPUDevice;
  packedSplats: PackedSplats;
  projectBindGroupLayout: GPUBindGroupLayout;
  renderBindGroupLayout: GPUBindGroupLayout;
  uniformByteLength: number;
}) {
  const wordsPerSplat = 4;
  const maxTextureDimension2D = device.limits?.maxTextureDimension2D ?? 8192;
  const textureWidth = Math.min(4096, maxTextureDimension2D);
  const maxSplatsPerChunk = textureWidth * maxTextureDimension2D;
  const packedArray = packedSplats.packedArray;
  if (!packedArray) {
    throw new Error("PackedSplats is not initialized");
  }
  const totalCount = packedSplats.getNumSplats();
  const chunks = [];
  for (let base = 0; base < totalCount; base += maxSplatsPerChunk) {
    const count = Math.min(maxSplatsPerChunk, totalCount - base);
    const textureHeight = Math.ceil(count / textureWidth);
    const textureSplatCount = textureWidth * textureHeight;
    const textureData = new Uint32Array(textureSplatCount * wordsPerSplat);
    textureData.set(
      packedArray.subarray(
        base * wordsPerSplat,
        (base + count) * wordsPerSplat,
      ),
    );
    const splatTexture = device.createTexture({
      label: "spark_wgsl_splat_texture",
      size: { width: textureWidth, height: textureHeight },
      format: "rgba32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: splatTexture },
      textureData,
      {
        bytesPerRow:
          textureWidth * wordsPerSplat * Uint32Array.BYTES_PER_ELEMENT,
        rowsPerImage: textureHeight,
      },
      { width: textureWidth, height: textureHeight },
    );
    const sh1Texture = createChunkUintTexture({
      device,
      label: "spark_wgsl_sh1_texture",
      format: "rg32uint",
      textureWidth,
      textureHeight,
      wordsPerTexel: 2,
      source: packedSplats.extra.sh1 as Uint32Array | undefined,
      sourceWordsPerSplat: 2,
      base,
      count,
    });
    const sh2Texture = createChunkUintTexture({
      device,
      label: "spark_wgsl_sh2_texture",
      format: "rgba32uint",
      textureWidth,
      textureHeight,
      wordsPerTexel: 4,
      source: packedSplats.extra.sh2 as Uint32Array | undefined,
      sourceWordsPerSplat: 4,
      base,
      count,
    });
    const sh3Texture = createChunkUintTexture({
      device,
      label: "spark_wgsl_sh3_texture",
      format: "rgba32uint",
      textureWidth,
      textureHeight,
      wordsPerTexel: 4,
      source: packedSplats.extra.sh3 as Uint32Array | undefined,
      sourceWordsPerSplat: 4,
      base,
      count,
    });
    const identityOrderData = createIdentityOrderData(
      base,
      count,
      textureWidth,
      textureHeight,
    );
    const orderTextures = [0, 1].map(() => {
      const orderTexture = device.createTexture({
        label: "spark_wgsl_order_texture",
        size: { width: textureWidth, height: textureHeight },
        format: "r32uint",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: orderTexture },
        identityOrderData,
        {
          bytesPerRow: textureWidth * Uint32Array.BYTES_PER_ELEMENT,
          rowsPerImage: textureHeight,
        },
        { width: textureWidth, height: textureHeight },
      );
      return orderTexture;
    }) as [GPUTexture, GPUTexture];
    const uniformData = new Float32Array(
      uniformByteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    uniformData[34] = count;
    uniformData[40] = textureWidth;
    const uniformBuffer = createBuffer(
      device,
      "spark_wgsl_uniforms",
      uniformData,
      GPUBufferUsage.UNIFORM,
    );
    const projectedBuffer = device.createBuffer({
      label: "spark_wgsl_projected_compact",
      size: count * 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    const projectedColorOpacityBuffer = device.createBuffer({
      label: "spark_wgsl_projected_color_opacity",
      size: count * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    const projectedCenterOpacityBuffer = device.createBuffer({
      label: "spark_wgsl_projected_center_opacity",
      size: count * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    const projectedAxesBuffer = device.createBuffer({
      label: "spark_wgsl_projected_axes",
      size: count * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
    });
    const projectBindGroups = orderTextures.map((orderTexture) =>
      device.createBindGroup({
        layout: projectBindGroupLayout,
        entries: [
          { binding: 0, resource: splatTexture.createView() },
          { binding: 1, resource: { buffer: uniformBuffer } },
          { binding: 2, resource: orderTexture.createView() },
          { binding: 3, resource: { buffer: projectedBuffer } },
          { binding: 4, resource: { buffer: projectedColorOpacityBuffer } },
          { binding: 5, resource: { buffer: projectedCenterOpacityBuffer } },
          { binding: 6, resource: { buffer: projectedAxesBuffer } },
          { binding: 7, resource: sh1Texture.createView() },
          { binding: 8, resource: sh2Texture.createView() },
          { binding: 9, resource: sh3Texture.createView() },
        ],
      }),
    ) as [GPUBindGroup, GPUBindGroup];
    const renderBindGroup = device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: projectedBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: projectedColorOpacityBuffer } },
        { binding: 3, resource: { buffer: projectedCenterOpacityBuffer } },
        { binding: 4, resource: { buffer: projectedAxesBuffer } },
      ],
    });
    chunks.push({
      base,
      count,
      drawCounts: [count, count] as [number, number],
      textureWidth,
      textureHeight,
      splatTexture,
      sh1Texture,
      sh2Texture,
      sh3Texture,
      orderTextures,
      projectedBuffer,
      projectedColorOpacityBuffer,
      projectedCenterOpacityBuffer,
      projectedAxesBuffer,
      identityOrderData,
      uniformBuffer,
      projectBindGroups,
      renderBindGroup,
    });
  }
  return chunks;
}

function createChunkUintTexture({
  device,
  label,
  format,
  textureWidth,
  textureHeight,
  wordsPerTexel,
  source,
  sourceWordsPerSplat,
  base,
  count,
}: {
  device: GPUDevice;
  label: string;
  format: GPUTextureFormat;
  textureWidth: number;
  textureHeight: number;
  wordsPerTexel: number;
  source?: Uint32Array;
  sourceWordsPerSplat: number;
  base: number;
  count: number;
}) {
  const width = source ? textureWidth : 1;
  const height = source ? textureHeight : 1;
  const texture = device.createTexture({
    label,
    size: { width, height },
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const textureSplatCount = width * height;
  const rowWords = width * wordsPerTexel;
  const alignedRowBytes = alignTo(
    rowWords * Uint32Array.BYTES_PER_ELEMENT,
    256,
  );
  const alignedRowWords = alignedRowBytes / Uint32Array.BYTES_PER_ELEMENT;
  const textureData = new Uint32Array(alignedRowWords * height);
  if (source) {
    const sourceOffset = base * sourceWordsPerSplat;
    for (let row = 0; row < textureHeight; row++) {
      const splatOffset = row * textureWidth;
      const rowSplatCount = Math.min(
        textureWidth,
        Math.max(0, count - splatOffset),
      );
      if (rowSplatCount <= 0) {
        break;
      }
      const sourceRowStart = sourceOffset + splatOffset * sourceWordsPerSplat;
      const sourceRowEnd = sourceRowStart + rowSplatCount * sourceWordsPerSplat;
      textureData.set(
        source.subarray(sourceRowStart, sourceRowEnd),
        row * alignedRowWords,
      );
    }
  }
  device.queue.writeTexture(
    { texture },
    textureData,
    {
      bytesPerRow: alignedRowBytes,
      rowsPerImage: height,
    },
    { width, height },
  );
  return texture;
}

function alignTo(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

function createBuffer(
  device: GPUDevice,
  label: string,
  data: Float32Array | Uint32Array,
  usage: GPUBufferUsageFlags,
) {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  if (data instanceof Float32Array) {
    new Float32Array(buffer.getMappedRange()).set(data);
  } else {
    new Uint32Array(buffer.getMappedRange()).set(data);
  }
  buffer.unmap();
  return buffer;
}

function nextPowerOfTwo(value: number) {
  return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}

function createIdentityOrderData(
  base: number,
  count: number,
  textureWidth: number,
  textureHeight: number,
) {
  const data = new Uint32Array(textureWidth * textureHeight);
  for (let i = 0; i < count; i++) {
    data[i] = base + i;
  }
  return data;
}

function createChunkOrderData(
  indices: Uint32Array,
  visibleCount: number,
  base: number,
  count: number,
  textureWidth: number,
) {
  const textureHeight = Math.ceil(count / textureWidth);
  const data = new Uint32Array(textureWidth * textureHeight);
  let out = 0;
  const end = base + count;
  for (let i = 0; i < visibleCount && out < count; i++) {
    const index = indices[i];
    if (index >= base && index < end) {
      data[out++] = index;
    }
  }
  return { data, count: out };
}

function createDirectWGSLSortData(
  packedSplats: PackedSplats,
): DirectWGSLSortData {
  const source: WebGPUSplatSource = packedSplats;
  const count = source.getNumSplats();
  const positions = new Float32Array(count * 3);
  const indices = new Uint32Array(count);
  source.forEachSplat((index, center) => {
    const i3 = index * 3;
    positions[i3] = center.x;
    positions[i3 + 1] = center.y;
    positions[i3 + 2] = center.z;
    indices[index] = index;
  });
  return { positions, indices, visibleCount: count };
}

const scratchSortMatrix = new THREE.Matrix4();
const scratchWorldInverse = new THREE.Matrix4();
const scratchOrderPosition = new THREE.Vector3();
const scratchOrderQuaternion = new THREE.Quaternion();
const scratchFlushPosition = new THREE.Vector3();
const scratchFlushQuaternion = new THREE.Quaternion();
const projectWorkgroupSize = 256;
const gpuSortWorkgroupSize = 256;
const gpuRadixBlockPrefixGroupSize = 256;
const gpuCountingScanBlockSize = 256;
const gpuCountingGroupScanBlockSize = 512;
const gpuRadixScatterWorkgroupBytes = 256 * 8 * Uint32Array.BYTES_PER_ELEMENT;
const defaultExperimentalGPUSortMaxSplats = 262144;
const batchedQuadSplatCount = 128;
const batchedQuadIndexCount = batchedQuadSplatCount * 6;
const defaultRenderStateMatrixEpsilon = 1e-3;
const renderStateScalarEpsilon = 1e-6;
const maxInFlightFrames = 2;

function clampGPUSortBucketBits(value: number) {
  if (!Number.isFinite(value)) {
    return 16;
  }
  const bits = Math.floor(value);
  if (bits <= 8) {
    return 8;
  }
  if (bits <= 16) {
    return 16;
  }
  if (bits <= 24) {
    return 24;
  }
  return 32;
}

function clampGPUCountingBucketCount(value: number) {
  if (!Number.isFinite(value)) {
    return 131072;
  }
  const bucketCount = Math.floor(value);
  if (bucketCount <= 1024) {
    return 1024;
  }
  if (bucketCount <= 2048) {
    return 2048;
  }
  if (bucketCount <= 4096) {
    return 4096;
  }
  if (bucketCount <= 8192) {
    return 8192;
  }
  if (bucketCount <= 16384) {
    return 16384;
  }
  if (bucketCount <= 32768) {
    return 32768;
  }
  if (bucketCount <= 65536) {
    return 65536;
  }
  return 131072;
}

function cloneCameraState(camera: THREE.Camera) {
  const clone = camera.clone();
  clone.matrix.copy(camera.matrix);
  clone.matrixWorld.copy(camera.matrixWorld);
  clone.matrixWorldInverse.copy(camera.matrixWorldInverse);
  clone.projectionMatrix.copy(camera.projectionMatrix);
  clone.projectionMatrixInverse.copy(camera.projectionMatrixInverse);
  clone.matrixAutoUpdate = false;
  clone.matrixWorldAutoUpdate = false;
  clone.matrixWorldNeedsUpdate = false;
  return clone;
}

function createBatchedQuadGeometry(device: GPUDevice) {
  const vertices = new Float32Array(batchedQuadSplatCount * 4 * 3);
  const indices = new Uint32Array(batchedQuadIndexCount);
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ];
  for (let splat = 0; splat < batchedQuadSplatCount; splat++) {
    const vertexBase = splat * 4;
    for (let corner = 0; corner < 4; corner++) {
      const offset = (vertexBase + corner) * 3;
      vertices[offset] = corners[corner][0];
      vertices[offset + 1] = corners[corner][1];
      vertices[offset + 2] = splat;
    }
    const indexOffset = splat * 6;
    indices[indexOffset] = vertexBase;
    indices[indexOffset + 1] = vertexBase + 1;
    indices[indexOffset + 2] = vertexBase + 2;
    indices[indexOffset + 3] = vertexBase;
    indices[indexOffset + 4] = vertexBase + 2;
    indices[indexOffset + 5] = vertexBase + 3;
  }
  return {
    vertexBuffer: createBuffer(
      device,
      "spark_wgsl_batched_quad_vertices",
      vertices,
      GPUBufferUsage.VERTEX,
    ),
    indexBuffer: createBuffer(
      device,
      "spark_wgsl_batched_quad_indices",
      indices,
      GPUBufferUsage.INDEX,
    ),
  };
}

const shaderCommonWGSL = /* wgsl */ `
struct Uniforms {
  localToView: mat4x4<f32>,
  projection: mat4x4<f32>,
  renderSizeCountScale: vec4<f32>,
  opacityAlphaRadiusStdDev: vec4<f32>,
  textureInfo: vec4<f32>,
  blurInfo: vec4<f32>,
  cameraInfo: vec4<f32>,
  shInfo: vec4<f32>,
};

fn quatVec(q: vec4<f32>, axis: vec3<f32>) -> vec3<f32> {
  let t = cross(q.xyz, axis) + axis * q.w;
  return axis + cross(q.xyz, t) * 2.0;
}

fn decodeQuatOctXy88R8(encoded: u32) -> vec4<f32> {
  let quantU = encoded & 0xffu;
  let quantV = (encoded >> 8u) & 0xffu;
  let angleInt = encoded >> 16u;
  let u = f32(quantU) / 255.0;
  let v = f32(quantV) / 255.0;
  var axis = vec3<f32>(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0);
  axis.z = 1.0 - abs(axis.x) - abs(axis.y);
  let t = max(-axis.z, 0.0);
  axis.x += select(t, -t, axis.x >= 0.0);
  axis.y += select(t, -t, axis.y >= 0.0);
  axis = normalize(axis);
  let theta = (f32(angleInt) / 255.0) * 3.141592653589793;
  let halfTheta = theta * 0.5;
  return vec4<f32>(axis * sin(halfTheta), cos(halfTheta));
}

struct DecodedSplat {
  center: vec3<f32>,
  scales: vec3<f32>,
  quat: vec4<f32>,
  color: vec3<f32>,
  opacity: f32,
};

fn unpackSplat(packed: vec4<u32>) -> DecodedSplat {
  let word0 = packed.x;
  let word1 = packed.y;
  let word2 = packed.z;
  let word3 = packed.w;
  var splat: DecodedSplat;
  splat.color = vec3<f32>(
    f32(word0 & 0xffu),
    f32((word0 >> 8u) & 0xffu),
    f32((word0 >> 16u) & 0xffu)
  ) / 255.0;
  splat.opacity = f32((word0 >> 24u) & 0xffu) / 255.0;
  let xy = unpack2x16float(word1);
  let z = unpack2x16float(word2 & 0xffffu).x;
  splat.center = vec3<f32>(xy, z);
  let lnScaleMin = -12.0;
  let lnScaleScale = (9.0 - lnScaleMin) / 254.0;
  let sx = word3 & 0xffu;
  let sy = (word3 >> 8u) & 0xffu;
  let sz = (word3 >> 16u) & 0xffu;
  splat.scales = vec3<f32>(
    select(0.0, exp(lnScaleMin + f32(sx - 1u) * lnScaleScale), sx != 0u),
    select(0.0, exp(lnScaleMin + f32(sy - 1u) * lnScaleScale), sy != 0u),
    select(0.0, exp(lnScaleMin + f32(sz - 1u) * lnScaleScale), sz != 0u)
  );
  let encodedQuat = ((word2 >> 16u) & 0xffffu) | ((word3 >> 8u) & 0xff0000u);
  splat.quat = decodeQuatOctXy88R8(encodedQuat);
  return splat;
}

fn evaluatePackedSH1(packedData: vec2<u32>, viewDir: vec3<f32>, sh1Max: f32) -> vec3<f32> {
  let sh1_0 = vec3<f32>(vec3<i32>(
    i32(packedData.x << 25u) >> 25,
    i32(packedData.x << 18u) >> 25,
    i32(packedData.x << 11u) >> 25
  ));
  let sh1_1 = vec3<f32>(vec3<i32>(
    i32(packedData.x << 4u) >> 25,
    i32((packedData.x >> 3u) | (packedData.y << 29u)) >> 25,
    i32(packedData.y << 22u) >> 25
  ));
  let sh1_2 = vec3<f32>(vec3<i32>(
    i32(packedData.y << 15u) >> 25,
    i32(packedData.y << 8u) >> 25,
    i32(packedData.y << 1u) >> 25
  ));
  let rgb = sh1_0 * (-0.4886025 * viewDir.y)
    + sh1_1 * (0.4886025 * viewDir.z)
    + sh1_2 * (-0.4886025 * viewDir.x);
  return rgb * (sh1Max / 63.0);
}

fn evaluatePackedSH2(packedData: vec4<u32>, viewDir: vec3<f32>, sh2Max: f32) -> vec3<f32> {
  let sh2_0 = vec3<f32>(vec3<i32>(
    i32(packedData.x << 24u) >> 24,
    i32(packedData.x << 16u) >> 24,
    i32(packedData.x << 8u) >> 24
  ));
  let sh2_1 = vec3<f32>(vec3<i32>(
    i32(packedData.x) >> 24,
    i32(packedData.y << 24u) >> 24,
    i32(packedData.y << 16u) >> 24
  ));
  let sh2_2 = vec3<f32>(vec3<i32>(
    i32(packedData.y << 8u) >> 24,
    i32(packedData.y) >> 24,
    i32(packedData.z << 24u) >> 24
  ));
  let sh2_3 = vec3<f32>(vec3<i32>(
    i32(packedData.z << 16u) >> 24,
    i32(packedData.z << 8u) >> 24,
    i32(packedData.z) >> 24
  ));
  let sh2_4 = vec3<f32>(vec3<i32>(
    i32(packedData.w << 24u) >> 24,
    i32(packedData.w << 16u) >> 24,
    i32(packedData.w << 8u) >> 24
  ));
  let rgb = sh2_0 * (1.0925484 * viewDir.x * viewDir.y)
    + sh2_1 * (-1.0925484 * viewDir.y * viewDir.z)
    + sh2_2 * (0.3153915 * (2.0 * viewDir.z * viewDir.z - viewDir.x * viewDir.x - viewDir.y * viewDir.y))
    + sh2_3 * (-1.0925484 * viewDir.x * viewDir.z)
    + sh2_4 * (0.5462742 * (viewDir.x * viewDir.x - viewDir.y * viewDir.y));
  return rgb * (sh2Max / 127.0);
}

fn evaluatePackedSH3(packedData: vec4<u32>, viewDir: vec3<f32>, sh3Max: f32) -> vec3<f32> {
  let sh3_0 = vec3<f32>(vec3<i32>(
    i32(packedData.x << 26u) >> 26,
    i32(packedData.x << 20u) >> 26,
    i32(packedData.x << 14u) >> 26
  ));
  let sh3_1 = vec3<f32>(vec3<i32>(
    i32(packedData.x << 8u) >> 26,
    i32(packedData.x << 2u) >> 26,
    i32((packedData.x >> 4u) | (packedData.y << 28u)) >> 26
  ));
  let sh3_2 = vec3<f32>(vec3<i32>(
    i32(packedData.y << 22u) >> 26,
    i32(packedData.y << 16u) >> 26,
    i32(packedData.y << 10u) >> 26
  ));
  let sh3_3 = vec3<f32>(vec3<i32>(
    i32(packedData.y << 4u) >> 26,
    i32((packedData.y >> 2u) | (packedData.z << 30u)) >> 26,
    i32(packedData.z << 24u) >> 26
  ));
  let sh3_4 = vec3<f32>(vec3<i32>(
    i32(packedData.z << 18u) >> 26,
    i32(packedData.z << 12u) >> 26,
    i32(packedData.z << 6u) >> 26
  ));
  let sh3_5 = vec3<f32>(vec3<i32>(
    i32(packedData.z) >> 26,
    i32(packedData.w << 26u) >> 26,
    i32(packedData.w << 20u) >> 26
  ));
  let sh3_6 = vec3<f32>(vec3<i32>(
    i32(packedData.w << 14u) >> 26,
    i32(packedData.w << 8u) >> 26,
    i32(packedData.w << 2u) >> 26
  ));
  let xx = viewDir.x * viewDir.x;
  let yy = viewDir.y * viewDir.y;
  let zz = viewDir.z * viewDir.z;
  let xy = viewDir.x * viewDir.y;
  let rgb = sh3_0 * (-0.5900436 * viewDir.y * (3.0 * xx - yy))
    + sh3_1 * (2.8906114 * xy * viewDir.z)
    + sh3_2 * (-0.4570458 * viewDir.y * (4.0 * zz - xx - yy))
    + sh3_3 * (0.3731763 * viewDir.z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
    + sh3_4 * (-0.4570458 * viewDir.x * (4.0 * zz - xx - yy))
    + sh3_5 * (1.4453057 * viewDir.z * (xx - yy))
    + sh3_6 * (-0.5900436 * viewDir.x * (xx - 3.0 * yy));
  return rgb * (sh3Max / 31.0);
}
`;

const projectWGSL = `${shaderCommonWGSL}
@group(0) @binding(0) var splatTexture: texture_2d<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var orderTexture: texture_2d<u32>;
@group(0) @binding(3) var<storage, read_write> projected: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> projectedColorOpacity: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> projectedCenterOpacity: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> projectedAxes: array<vec4<f32>>;
@group(0) @binding(7) var sh1Texture: texture_2d<u32>;
@group(0) @binding(8) var sh2Texture: texture_2d<u32>;
@group(0) @binding(9) var sh3Texture: texture_2d<u32>;

fn packProjected(center: vec2<f32>, axis1: vec2<f32>, axis2: vec2<f32>, color: vec3<f32>, opacity: f32) -> vec4<u32> {
  return vec4<u32>(
    pack2x16float(center),
    pack2x16float(axis1),
    pack2x16float(axis2),
    (u32(clamp(color.r, 0.0, 1.0) * 255.0) & 0xffu) |
      ((u32(clamp(color.g, 0.0, 1.0) * 255.0) & 0xffu) << 8u) |
      ((u32(clamp(color.b, 0.0, 1.0) * 255.0) & 0xffu) << 16u) |
      ((u32(clamp(opacity, 0.0, 1.0) * 255.0) & 0xffu) << 24u)
  );
}

@compute @workgroup_size(${projectWorkgroupSize})
fn cs_main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let instanceIndex = globalId.x;
  if (instanceIndex >= u32(uniforms.renderSizeCountScale.z)) {
    return;
  }

  let textureWidth = u32(uniforms.textureInfo.x);
  let chunkBase = u32(uniforms.textureInfo.y);
  let sourceIndex = textureLoad(
    orderTexture,
    vec2<i32>(i32(instanceIndex % textureWidth), i32(instanceIndex / textureWidth)),
    0
  ).x;
  let localIndex = sourceIndex - chunkBase;
  let packed = textureLoad(
    splatTexture,
    vec2<i32>(i32(localIndex % textureWidth), i32(localIndex / textureWidth)),
    0
  );
  let splat = unpackSplat(packed);
  var color = splat.color;
  let numSh = u32(uniforms.shInfo.x);
  if (numSh > 0u) {
    let coord = vec2<i32>(i32(localIndex % textureWidth), i32(localIndex / textureWidth));
    let viewDir = normalize(splat.center - uniforms.cameraInfo.xyz);
    color = color + evaluatePackedSH1(textureLoad(sh1Texture, coord, 0).rg, viewDir, uniforms.shInfo.y);
    if (numSh >= 2u) {
      color = color + evaluatePackedSH2(textureLoad(sh2Texture, coord, 0), viewDir, uniforms.shInfo.z);
    }
    if (numSh >= 3u) {
      color = color + evaluatePackedSH3(textureLoad(sh3Texture, coord, 0), viewDir, uniforms.shInfo.w);
    }
  }

  let opacity = splat.opacity * uniforms.opacityAlphaRadiusStdDev.x;
  let minAlpha = uniforms.opacityAlphaRadiusStdDev.y;
  if (opacity < minAlpha) {
    projected[instanceIndex] = vec4<u32>(0u);
    projectedColorOpacity[instanceIndex] = vec4<f32>(0.0);
    projectedCenterOpacity[instanceIndex] = vec4<f32>(0.0);
    return;
  }

  let viewCenter = uniforms.localToView * vec4<f32>(splat.center, 1.0);
  if (viewCenter.z >= 0.0) {
    projected[instanceIndex] = vec4<u32>(0u);
    projectedColorOpacity[instanceIndex] = vec4<f32>(0.0);
    projectedCenterOpacity[instanceIndex] = vec4<f32>(0.0);
    return;
  }
  let clipCenter = uniforms.projection * viewCenter;
  if (abs(clipCenter.z) >= clipCenter.w) {
    projected[instanceIndex] = vec4<u32>(0u);
    projectedColorOpacity[instanceIndex] = vec4<f32>(0.0);
    projectedCenterOpacity[instanceIndex] = vec4<f32>(0.0);
    return;
  }
  let clipXY = clipCenter.w * 1.4;
  if (abs(clipCenter.x) > clipXY || abs(clipCenter.y) > clipXY) {
    projected[instanceIndex] = vec4<u32>(0u);
    projectedColorOpacity[instanceIndex] = vec4<f32>(0.0);
    projectedCenterOpacity[instanceIndex] = vec4<f32>(0.0);
    return;
  }

  let sizeScale = uniforms.renderSizeCountScale.w;
  let scale = splat.scales * sizeScale;
  let axisX = (uniforms.localToView * vec4<f32>(quatVec(splat.quat, vec3<f32>(1.0, 0.0, 0.0)) * scale.x, 0.0)).xyz;
  let axisY = (uniforms.localToView * vec4<f32>(quatVec(splat.quat, vec3<f32>(0.0, 1.0, 0.0)) * scale.y, 0.0)).xyz;
  let axisZ = (uniforms.localToView * vec4<f32>(quatVec(splat.quat, vec3<f32>(0.0, 0.0, 1.0)) * scale.z, 0.0)).xyz;

  let focal = 0.5 * uniforms.renderSizeCountScale.xy * vec2<f32>(uniforms.projection[0][0], uniforms.projection[1][1]);
  let invZ = 1.0 / viewCenter.z;
  let j1 = focal * invZ;
  let j2 = -(j1 * viewCenter.xy) * invZ;
  let covAxisX = vec2<f32>(j1.x * axisX.x + j2.x * axisX.z, j1.y * axisX.y + j2.y * axisX.z);
  let covAxisY = vec2<f32>(j1.x * axisY.x + j2.x * axisY.z, j1.y * axisY.y + j2.y * axisY.z);
  let covAxisZ = vec2<f32>(j1.x * axisZ.x + j2.x * axisZ.z, j1.y * axisZ.y + j2.y * axisZ.z);
  let covA0 = dot(vec3<f32>(covAxisX.x, covAxisY.x, covAxisZ.x), vec3<f32>(covAxisX.x, covAxisY.x, covAxisZ.x));
  let covD0 = dot(vec3<f32>(covAxisX.y, covAxisY.y, covAxisZ.y), vec3<f32>(covAxisX.y, covAxisY.y, covAxisZ.y));
  let covB = dot(vec3<f32>(covAxisX.x, covAxisY.x, covAxisZ.x), vec3<f32>(covAxisX.y, covAxisY.y, covAxisZ.y));

  let preBlurAmount = uniforms.blurInfo.y;
  let covA = covA0 + preBlurAmount;
  let covD = covD0 + preBlurAmount;
  let detOrig = max(covA * covD - covB * covB, 0.0);
  let blurAmount = uniforms.blurInfo.x;
  let a = covA + blurAmount;
  let d = covD + blurAmount;
  let det = max(a * d - covB * covB, 0.000001);
  let adjustedOpacity = opacity * sqrt(max(detOrig / det, 0.0));
  if (adjustedOpacity < minAlpha) {
    projected[instanceIndex] = vec4<u32>(0u);
    projectedColorOpacity[instanceIndex] = vec4<f32>(0.0);
    projectedCenterOpacity[instanceIndex] = vec4<f32>(0.0);
    return;
  }

  let eigenAvg = 0.5 * (a + d);
  let eigenDelta = sqrt(max(eigenAvg * eigenAvg - det, 0.0));
  let eigen1 = max(eigenAvg + eigenDelta, 0.0);
  let eigen2 = max(eigenAvg - eigenDelta, 0.0);
  let majorAxis = select(
    select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), a >= d),
    normalize(vec2<f32>(covB, eigen1 - a)),
    abs(covB) > 0.001
  );
  let minorAxis = vec2<f32>(majorAxis.y, -majorAxis.x);
  let maxPixelRadius = uniforms.opacityAlphaRadiusStdDev.z;
  let maxStdDev = uniforms.opacityAlphaRadiusStdDev.w;
  let scale1 = min(maxPixelRadius, maxStdDev * sqrt(eigen1));
  let scale2 = min(maxPixelRadius, maxStdDev * sqrt(eigen2));
  let ndcScale = vec2<f32>(2.0) / uniforms.renderSizeCountScale.xy;
  let ndcCenter = clipCenter.xyz / clipCenter.w;

  projected[instanceIndex] = packProjected(
    ndcCenter.xy,
    majorAxis * scale1 * ndcScale,
    minorAxis * scale2 * ndcScale,
    color,
    adjustedOpacity
  );
  projectedColorOpacity[instanceIndex] = vec4<f32>(color, adjustedOpacity);
  projectedCenterOpacity[instanceIndex] = vec4<f32>(ndcCenter.xy, adjustedOpacity, 1.0);
  projectedAxes[instanceIndex] = vec4<f32>(majorAxis * scale1 * ndcScale, minorAxis * scale2 * ndcScale);
}
`;

const gpuSortWGSL = /* wgsl */ `
struct SortUniforms {
  localToView0: vec4<f32>,
  localToView1: vec4<f32>,
  localToView2: vec4<f32>,
  localToView3: vec4<f32>,
  count: u32,
  paddedCount: u32,
  radial: u32,
  k: u32,
  j: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> pairs: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> uniforms: SortUniforms;
@group(0) @binding(3) var<storage, read_write> order: array<u32>;

fn sortableFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x80000000u) != 0u) {
    return ~bits;
  }
  return bits ^ 0x80000000u;
}

fn depthKey(index: u32) -> u32 {
  if (index >= uniforms.count) {
      return 0xffffffffu;
    }
  let i3 = index * 3u;
  let p = vec3<f32>(positions[i3], positions[i3 + 1u], positions[i3 + 2u]);
  let view = uniforms.localToView0 * p.x +
    uniforms.localToView1 * p.y +
    uniforms.localToView2 * p.z +
    uniforms.localToView3;
  let viewX = view.x;
  let viewY = view.y;
  let viewZ = view.z;
  if (uniforms.radial != 0u) {
    return sortableFloatKey(-(viewX * viewX + viewY * viewY + viewZ * viewZ));
  }
  return sortableFloatKey(viewZ);
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn make_keys(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.paddedCount) {
    return;
  }
  pairs[index] = vec2<u32>(depthKey(index), index);
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn bitonic_pass(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let i = globalId.x;
  if (i >= uniforms.paddedCount) {
    return;
  }
  let ixj = i ^ uniforms.j;
  if (ixj <= i || ixj >= uniforms.paddedCount) {
    return;
  }
  let a = pairs[i];
  let b = pairs[ixj];
  let ascending = (i & uniforms.k) == 0u;
  let aBeforeB = (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
  let aAfterB = (a.x > b.x) || ((a.x == b.x) && (a.y > b.y));
  let shouldSwap = select(aBeforeB, aAfterB, ascending);
  if (shouldSwap) {
    pairs[i] = b;
    pairs[ixj] = a;
  }
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn extract_order(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.count) {
    return;
  }
  order[index] = pairs[index].y;
}
`;

const gpuRadixSortWGSL = /* wgsl */ `
struct RadixUniforms {
  localToView0: vec4<f32>,
  localToView1: vec4<f32>,
  localToView2: vec4<f32>,
  localToView3: vec4<f32>,
  count: u32,
  shift: u32,
  radial: u32,
  passIndex: u32,
  blockCount: u32,
  blockGroupCount: u32,
  keyBits: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> pairsA: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> pairsB: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> order: array<u32>;
@group(0) @binding(4) var<storage, read_write> histogramData: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> prefixData: array<u32>;
@group(0) @binding(6) var<storage, read_write> bucketPrefixData: array<u32>;
@group(0) @binding(7) var<uniform> uniforms: RadixUniforms;

var<workgroup> scatterBucketMasks: array<atomic<u32>, 2048>;

fn sortableFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x80000000u) != 0u) {
    return ~bits;
  }
  return bits ^ 0x80000000u;
}

fn depthKey(index: u32) -> u32 {
  if (index >= uniforms.count) {
      return 0xffffffffu;
    }
  let i3 = index * 3u;
  let p = vec3<f32>(positions[i3], positions[i3 + 1u], positions[i3 + 2u]);
  let view = uniforms.localToView0 * p.x +
    uniforms.localToView1 * p.y +
    uniforms.localToView2 * p.z +
    uniforms.localToView3;
  if (uniforms.radial != 0u) {
    return sortableFloatKey(-(view.x * view.x + view.y * view.y + view.z * view.z));
  }
  return sortableFloatKey(view.z);
}

fn quantizeDepthKey(key: u32) -> u32 {
  if (uniforms.keyBits >= 32u) {
    return key;
  }
  let bits = max(8u, uniforms.keyBits);
  let shift = 32u - bits;
  return key >> shift;
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn make_keys(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.count) {
    return;
  }
  pairsA[index] = vec2<u32>(quantizeDepthKey(depthKey(index)), index);
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn histogram(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let index = globalId.x;
  let localIndex = localId.x;
  let block = workgroupId.x;
  if (localIndex < 256u) {
    atomicStore(&histogramData[block * 256u + localIndex], 0u);
  }
  storageBarrier();
  if (index >= uniforms.count) {
    return;
  }
  let pair = select(pairsA[index], pairsB[index], (uniforms.passIndex & 1u) != 0u);
  let bucket = (pair.x >> uniforms.shift) & 0xffu;
  atomicAdd(&histogramData[block * 256u + bucket], 1u);
}

@compute @workgroup_size(256)
fn bucket_totals(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let bucket = localId.x;
  let group = workgroupId.y;
  if (bucket >= 256u) {
    return;
  }
  if (group >= uniforms.blockGroupCount) {
    return;
  }
  let firstBlock = group * ${gpuRadixBlockPrefixGroupSize}u;
  let endBlock = min(firstBlock + ${gpuRadixBlockPrefixGroupSize}u, uniforms.blockCount);
  var total = 0u;
  for (var block = firstBlock; block < endBlock; block = block + 1u) {
    total = total + atomicLoad(&histogramData[block * 256u + bucket]);
  }
  bucketPrefixData[512u + bucket * uniforms.blockGroupCount + group] = total;
}

@compute @workgroup_size(1)
fn bucket_bases() {
  var bucketBase = 0u;
  for (var bucket = 0u; bucket < 256u; bucket = bucket + 1u) {
    bucketPrefixData[256u + bucket] = bucketBase;
    var groupBase = bucketBase;
    for (var group = 0u; group < uniforms.blockGroupCount; group = group + 1u) {
      let offset = 512u + bucket * uniforms.blockGroupCount + group;
      let total = bucketPrefixData[offset];
      bucketPrefixData[offset] = groupBase;
      groupBase = groupBase + total;
    }
    bucketPrefixData[bucket] = groupBase - bucketBase;
    bucketBase = groupBase;
  }
}

@compute @workgroup_size(256)
fn block_prefix(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let bucket = localId.x;
  let group = workgroupId.y;
  if (bucket >= 256u) {
    return;
  }
  if (group >= uniforms.blockGroupCount) {
    return;
  }
  let firstBlock = group * ${gpuRadixBlockPrefixGroupSize}u;
  let endBlock = min(firstBlock + ${gpuRadixBlockPrefixGroupSize}u, uniforms.blockCount);
  var blockBase = bucketPrefixData[512u + bucket * uniforms.blockGroupCount + group];
  for (var block = firstBlock; block < endBlock; block = block + 1u) {
    let offset = block * 256u + bucket;
    let count = atomicLoad(&histogramData[offset]);
    prefixData[offset] = blockBase;
    blockBase = blockBase + count;
  }
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn scatter(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let index = globalId.x;
  let localIndex = localId.x;
  for (var clearIndex = localIndex; clearIndex < 2048u; clearIndex = clearIndex + ${gpuSortWorkgroupSize}u) {
    atomicStore(&scatterBucketMasks[clearIndex], 0u);
  }
  workgroupBarrier();

  var pair = vec2<u32>(0xffffffffu, 0xffffffffu);
  var bucket = 0u;
  let isActive = index < uniforms.count;
  if (isActive) {
    pair = select(pairsA[index], pairsB[index], (uniforms.passIndex & 1u) != 0u);
    bucket = (pair.x >> uniforms.shift) & 0xffu;
    let word = localIndex >> 5u;
    let bit = 1u << (localIndex & 31u);
    atomicOr(&scatterBucketMasks[bucket * 8u + word], bit);
  }
  workgroupBarrier();

  if (isActive) {
    let word = localIndex >> 5u;
    var localRank = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
      if (i < word) {
        localRank = localRank + countOneBits(atomicLoad(&scatterBucketMasks[bucket * 8u + i]));
      }
    }
    if (word < 8u) {
      let beforeMask = (1u << (localIndex & 31u)) - 1u;
      localRank = localRank + countOneBits(atomicLoad(&scatterBucketMasks[bucket * 8u + word]) & beforeMask);
    }
    let outIndex = prefixData[workgroupId.x * 256u + bucket] + localRank;
    if ((uniforms.passIndex & 1u) == 0u) {
      pairsB[outIndex] = pair;
    } else {
      pairsA[outIndex] = pair;
    }
  }
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn extract_order(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.count) {
    return;
  }
  let pair = select(pairsA[index], pairsB[index], (uniforms.passIndex & 1u) != 0u);
  order[index] = pair.y;
}
`;

const gpuCountingSortWGSL = /* wgsl */ `
struct CountingUniforms {
  localToView0: vec4<f32>,
  localToView1: vec4<f32>,
  localToView2: vec4<f32>,
  localToView3: vec4<f32>,
  count: u32,
  bucketCount: u32,
  radial: u32,
  bucketShift: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> histogramData: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> offsetsData: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> groupTotalsData: array<u32>;
@group(0) @binding(6) var<uniform> uniforms: CountingUniforms;

var<workgroup> scanData: array<u32, ${gpuCountingScanBlockSize}>;
var<workgroup> groupScanData: array<u32, ${gpuCountingGroupScanBlockSize}>;

fn sortableFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x80000000u) != 0u) {
    return ~bits;
  }
  return bits ^ 0x80000000u;
}

fn depthKey(index: u32) -> u32 {
  let i3 = index * 3u;
  let p = vec3<f32>(positions[i3], positions[i3 + 1u], positions[i3 + 2u]);
  let view = uniforms.localToView0 * p.x +
    uniforms.localToView1 * p.y +
    uniforms.localToView2 * p.z +
    uniforms.localToView3;
  if (uniforms.radial != 0u) {
    return sortableFloatKey(-(view.x * view.x + view.y * view.y + view.z * view.z));
  }
  return sortableFloatKey(view.z);
}

fn bucketForKey(key: u32) -> u32 {
  return min(key >> uniforms.bucketShift, uniforms.bucketCount - 1u);
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn key_histogram(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.count) {
    return;
  }
  let bucket = bucketForKey(depthKey(index));
  keys[index] = bucket;
  atomicAdd(&histogramData[bucket], 1u);
}

@compute @workgroup_size(${gpuCountingScanBlockSize})
fn block_prefix(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let localIndex = localId.x;
  let bucket = workgroupId.x * ${gpuCountingScanBlockSize}u + localIndex;
  var value = 0u;
  if (bucket < uniforms.bucketCount) {
    value = atomicLoad(&histogramData[bucket]);
  }
  scanData[localIndex] = value;
  workgroupBarrier();

  var offset = 1u;
  for (var d = ${gpuCountingScanBlockSize >> 1}u; d > 0u; d = d >> 1u) {
    if (localIndex < d) {
      let ai = offset * ((localIndex << 1u) + 1u) - 1u;
      let bi = offset * ((localIndex << 1u) + 2u) - 1u;
      scanData[bi] = scanData[bi] + scanData[ai];
    }
    offset = offset << 1u;
    workgroupBarrier();
  }

  if (localIndex == 0u) {
    groupTotalsData[workgroupId.x] = scanData[${gpuCountingScanBlockSize - 1}u];
    scanData[${gpuCountingScanBlockSize - 1}u] = 0u;
  }
  workgroupBarrier();

  for (var d = 1u; d < ${gpuCountingScanBlockSize}u; d = d << 1u) {
    offset = offset >> 1u;
    if (localIndex < d) {
      let ai = offset * ((localIndex << 1u) + 1u) - 1u;
      let bi = offset * ((localIndex << 1u) + 2u) - 1u;
      let t = scanData[ai];
      scanData[ai] = scanData[bi];
      scanData[bi] = scanData[bi] + t;
    }
    workgroupBarrier();
  }

  if (bucket < uniforms.bucketCount) {
    atomicStore(&offsetsData[bucket], scanData[localIndex]);
  }
}

@compute @workgroup_size(${gpuCountingScanBlockSize})
fn group_prefix(@builtin(local_invocation_id) localId: vec3<u32>) {
  let localIndex = localId.x;
  let groupCount = uniforms.bucketCount / ${gpuCountingScanBlockSize}u;
  let secondIndex = localIndex + ${gpuCountingScanBlockSize}u;
  var value0 = 0u;
  var value1 = 0u;
  if (localIndex < groupCount) {
    value0 = groupTotalsData[localIndex];
  }
  if (secondIndex < groupCount) {
    value1 = groupTotalsData[secondIndex];
  }
  groupScanData[localIndex] = value0;
  groupScanData[secondIndex] = value1;
  workgroupBarrier();

  var offset = 1u;
  for (var d = ${gpuCountingGroupScanBlockSize >> 1}u; d > 0u; d = d >> 1u) {
    if (localIndex < d) {
      let ai = offset * ((localIndex << 1u) + 1u) - 1u;
      let bi = offset * ((localIndex << 1u) + 2u) - 1u;
      groupScanData[bi] = groupScanData[bi] + groupScanData[ai];
    }
    offset = offset << 1u;
    workgroupBarrier();
  }

  if (localIndex == 0u) {
    groupScanData[${gpuCountingGroupScanBlockSize - 1}u] = 0u;
  }
  workgroupBarrier();

  for (var d = 1u; d < ${gpuCountingGroupScanBlockSize}u; d = d << 1u) {
    offset = offset >> 1u;
    if (localIndex < d) {
      let ai = offset * ((localIndex << 1u) + 1u) - 1u;
      let bi = offset * ((localIndex << 1u) + 2u) - 1u;
      let t = groupScanData[ai];
      groupScanData[ai] = groupScanData[bi];
      groupScanData[bi] = groupScanData[bi] + t;
    }
    workgroupBarrier();
  }

  if (localIndex < groupCount) {
    groupTotalsData[localIndex] = groupScanData[localIndex];
  }
  if (secondIndex < groupCount) {
    groupTotalsData[secondIndex] = groupScanData[secondIndex];
  }
}

@compute @workgroup_size(${gpuCountingScanBlockSize})
fn add_group_base(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
) {
  let localIndex = localId.x;
  let bucket = workgroupId.x * ${gpuCountingScanBlockSize}u + localIndex;
  if (bucket < uniforms.bucketCount) {
    let base = groupTotalsData[workgroupId.x];
    atomicStore(&offsetsData[bucket], atomicLoad(&offsetsData[bucket]) + base);
  }
}

@compute @workgroup_size(${gpuSortWorkgroupSize})
fn scatter(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= uniforms.count) {
    return;
  }
  let bucket = keys[index];
  let outIndex = atomicAdd(&offsetsData[bucket], 1u);
  order[outIndex] = index;
}
`;

const renderWGSL = `${shaderCommonWGSL}
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) opacityStdDev: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> projected: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> projectedColorOpacity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> projectedCenterOpacity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> projectedAxes: array<vec4<f32>>;

@vertex
fn vs_main(@location(0) batchVertex: vec3<f32>, @builtin(instance_index) instanceIndex: u32) -> VertexOut {
  let c = batchVertex.xy;
  let splatIndex = instanceIndex * ${batchedQuadSplatCount}u + u32(batchVertex.z);
  var out: VertexOut;
  let maxStdDev = uniforms.opacityAlphaRadiusStdDev.w;
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  out.uv = c * maxStdDev;
  out.color = vec3<f32>(0.0);
  out.opacityStdDev = vec2<f32>(0.0, maxStdDev);

  if (splatIndex >= u32(uniforms.renderSizeCountScale.z)) {
    return out;
  }

  let p = projected[splatIndex];
  if (p.w == 0u) {
    return out;
  }

  let colorOpacity = projectedColorOpacity[splatIndex];
  let color = colorOpacity.rgb;
  var opacity = colorOpacity.a;
  var center = unpack2x16float(p.x);
  var axis1 = unpack2x16float(p.y);
  var axis2 = unpack2x16float(p.z);
  if (uniforms.textureInfo.z > 0.5) {
    let centerOpacity = projectedCenterOpacity[splatIndex];
    let axes = projectedAxes[splatIndex];
    center = centerOpacity.xy;
    axis1 = axes.xy;
    axis2 = axes.zw;
    opacity = centerOpacity.z;
  }
  let ndc = center + axis1 * c.x + axis2 * c.y;

  out.position = vec4<f32>(ndc, 0.0, 1.0);
  out.color = color;
  out.opacityStdDev = vec2<f32>(opacity, maxStdDev);
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let z2 = dot(in.uv, in.uv);
  let alphaRadiusScale = max(uniforms.cameraInfo.w, 0.000001);
  let alphaStdDev = in.opacityStdDev.y * alphaRadiusScale;
  let maxStdDev2 = max(alphaStdDev * alphaStdDev, 0.000001);
  let a = z2 / maxStdDev2;
  if (a > 1.0) {
    discard;
  }
  let alphaBias = max(uniforms.blurInfo.w, 0.000001);
  let edge = exp(-alphaBias);
  let gaussian = (exp(-alphaBias * a) - edge) / (1.0 - edge);
  let alpha = in.opacityStdDev.x * max(gaussian, 0.0);
  if (alpha < uniforms.opacityAlphaRadiusStdDev.y) {
    discard;
  }
  return vec4<f32>(in.color * alpha, alpha);
}
`;
