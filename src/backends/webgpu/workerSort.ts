const directWGSLSortPositions = new Map<number, Float32Array>();

export const directWGSLWorkerSortHandlers = {
  initDirectWGSLSort,
  disposeDirectWGSLSort,
  sortDirectWGSLSplats,
};

function initDirectWGSLSort({
  id,
  positions,
}: {
  id: number;
  positions: Float32Array;
}) {
  directWGSLSortPositions.set(id, positions);
  return { id };
}

function disposeDirectWGSLSort({ id }: { id: number }) {
  directWGSLSortPositions.delete(id);
  return { id };
}

function sortDirectWGSLSplats({
  id,
  matrix,
  radial,
}: {
  id: number;
  matrix: Float32Array;
  radial: boolean;
}) {
  const positions = directWGSLSortPositions.get(id);
  if (!positions) {
    throw new Error(`Missing direct WGSL sort positions for id ${id}`);
  }
  const result = sortDirectWGSLPositions({
    positions,
    matrix,
    radial,
  });
  return result;
}

export function sortDirectWGSLPositions({
  positions,
  matrix,
  radial,
}: {
  positions: Float32Array;
  matrix: Float32Array;
  radial: boolean;
}) {
  const count = Math.floor(positions.length / 3);
  const ordering = new Uint32Array(count);
  const keys = new Uint32Array(count);
  let src = new Uint32Array(count);
  let dst = new Uint32Array(count);
  const keyFloat = new Float32Array(1);
  const keyBits = new Uint32Array(keyFloat.buffer);
  const counts = new Uint32Array(256);
  const e2 = matrix[2];
  const e6 = matrix[6];
  const e10 = matrix[10];
  const e14 = matrix[14];
  if (radial) {
    const e0 = matrix[0];
    const e4 = matrix[4];
    const e8 = matrix[8];
    const e12 = matrix[12];
    const e1 = matrix[1];
    const e5 = matrix[5];
    const e9 = matrix[9];
    const e13 = matrix[13];
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const x = positions[i3];
      const y = positions[i3 + 1];
      const z = positions[i3 + 2];
      const viewX = e0 * x + e4 * y + e8 * z + e12;
      const viewY = e1 * x + e5 * y + e9 * z + e13;
      const viewZ = e2 * x + e6 * y + e10 * z + e14;
      keyFloat[0] = -(viewX * viewX + viewY * viewY + viewZ * viewZ);
      const bits = keyBits[0];
      keys[i] =
        (bits & 0x80000000) !== 0 ? ~bits >>> 0 : (bits ^ 0x80000000) >>> 0;
      src[i] = i;
    }
  } else {
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      keyFloat[0] =
        e2 * positions[i3] +
        e6 * positions[i3 + 1] +
        e10 * positions[i3 + 2] +
        e14;
      const bits = keyBits[0];
      keys[i] =
        (bits & 0x80000000) !== 0 ? ~bits >>> 0 : (bits ^ 0x80000000) >>> 0;
      src[i] = i;
    }
  }
  for (let shift = 0; shift < 32; shift += 8) {
    counts.fill(0);
    for (let i = 0; i < count; i++) {
      counts[(keys[src[i]] >>> shift) & 0xff]++;
    }
    let sum = 0;
    for (let b = 0; b < 256; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < count; i++) {
      const index = src[i];
      const bucket = (keys[index] >>> shift) & 0xff;
      dst[counts[bucket]++] = index;
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }
  ordering.set(src);
  return { activeSplats: count, ordering };
}
