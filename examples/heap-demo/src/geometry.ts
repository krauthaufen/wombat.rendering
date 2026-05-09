// Pure procedural geometry — copied trim from wombat.dom's
// `scene/primitives/geometry.ts`. Just box / sphere / cylinder; the
// scene-graph layer would normally consume these via the cached
// shared-geometry handles in `scene/primitives/shared.ts`, but the
// demo builds RenderObjects by hand so we don't need any of that
// machinery.
//
// Each builder emits raw positions/normals/indices ready to drop
// straight into a `BufferView.ofArray`.

export interface GeometryData {
  readonly positions: Float32Array;
  readonly normals:   Float32Array;
  /** V2f per vertex; texture-mapping coordinates in [0,1]². */
  readonly uvs:       Float32Array;
  readonly indices:   Uint32Array;
}

const PI = Math.PI;
const TWO_PI = 2 * PI;

function pushV3(out: Float32Array, oi: number, v: readonly [number, number, number]): void {
  out[oi] = v[0]; out[oi + 1] = v[1]; out[oi + 2] = v[2];
}

// ---- Box: unit cube [0,1]^3 with split per-face normals (36 verts) ----

export function buildBox(): GeometryData {
  const O = 0, I = 1;
  const v = (x: number, y: number, z: number): [number, number, number] => [x, y, z];
  const verts: [number, number, number][] = [
    v(O,O,O), v(O,I,O), v(I,O,O), v(I,O,O), v(O,I,O), v(I,I,O),
    v(O,O,I), v(I,O,I), v(O,I,I), v(O,I,I), v(I,O,I), v(I,I,I),
    v(O,O,O), v(I,O,O), v(O,O,I), v(O,O,I), v(I,O,O), v(I,O,I),
    v(O,I,O), v(O,I,I), v(I,I,O), v(I,I,O), v(O,I,I), v(I,I,I),
    v(O,O,O), v(O,O,I), v(O,I,O), v(O,I,O), v(O,O,I), v(O,I,I),
    v(I,O,O), v(I,I,O), v(I,O,I), v(I,O,I), v(I,I,O), v(I,I,I),
  ];
  const ns: [number, number, number][] = [
    [0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1],
    [0,0, 1],[0,0, 1],[0,0, 1],[0,0, 1],[0,0, 1],[0,0, 1],
    [0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],
    [0, 1,0],[0, 1,0],[0, 1,0],[0, 1,0],[0, 1,0],[0, 1,0],
    [-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],
    [ 1,0,0],[ 1,0,0],[ 1,0,0],[ 1,0,0],[ 1,0,0],[ 1,0,0],
  ];
  // Per-face UV mapping. Six faces × six vertices each. The triangle
  // winding differs per face so two patterns repeat: A for faces where
  // the second triangle goes top-left → bot-right → top-right, B for
  // the mirror order.
  const A: ReadonlyArray<[number, number]> = [[0,0],[0,1],[1,0],[1,0],[0,1],[1,1]];
  const B: ReadonlyArray<[number, number]> = [[0,0],[1,0],[0,1],[0,1],[1,0],[1,1]];
  const facePat = [A, B, B, A, A, B];           // -Z, +Z, -Y, +Y, -X, +X
  const uvs = new Float32Array(verts.length * 2);
  const positions = new Float32Array(verts.length * 3);
  const normals   = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    pushV3(positions, i*3, verts[i]!);
    pushV3(normals, i*3, ns[i]!);
    const uv = facePat[Math.floor(i / 6)]![i % 6]!;
    uvs[i*2] = uv[0]; uvs[i*2 + 1] = uv[1];
  }
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, uvs, indices };
}

// ---- Sphere: unit sphere centred at origin ----

export function buildSphere(tessellation: number): GeometryData {
  const h = Math.floor(tessellation / 2);
  const dPhi = TWO_PI / tessellation;
  const dTheta = PI / h;
  const vertCount = (tessellation + 1) * (h - 1) + 2 * tessellation;
  const positions = new Float32Array(vertCount * 3);
  const normals   = new Float32Array(vertCount * 3);
  // Standard lat/long UV. Bands: u = x / tessellation, v = y / h.
  // Poles: u = (i + 0.5) / tessellation, v = 0 (bottom) or 1 (top).
  // Seam handled by the (tessellation + 1) verts per band — x=0 and
  // x=tessellation share position but get u=0 and u=1 respectively.
  const uvs = new Float32Array(vertCount * 2);
  let oi = 0;
  let theta = -PI / 2 + dTheta;
  for (let y = 1; y < h; y++) {
    let phi = 0;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let x = 0; x <= tessellation; x++) {
      const v: [number, number, number] = [Math.cos(phi)*ct, Math.sin(phi)*ct, st];
      pushV3(positions, oi*3, v); pushV3(normals, oi*3, v);
      uvs[oi*2] = x / tessellation; uvs[oi*2 + 1] = y / h;
      oi++;
      phi += dPhi;
    }
    theta += dTheta;
  }
  const n = oi;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi*3, [0,0, 1]); pushV3(normals, oi*3, [0,0, 1]);
    uvs[oi*2] = (i + 0.5) / tessellation; uvs[oi*2 + 1] = 1;
    oi++;
  }
  const sIdx = oi;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi*3, [0,0,-1]); pushV3(normals, oi*3, [0,0,-1]);
    uvs[oi*2] = (i + 0.5) / tessellation; uvs[oi*2 + 1] = 0;
    oi++;
  }

  const faces = 3 * (2 * tessellation + 2 * (h - 2) * tessellation);
  const indices = new Uint32Array(faces);
  let ii = 0;
  for (let x = 0; x < tessellation; x++) {
    indices[ii++] = x; indices[ii++] = sIdx + x; indices[ii++] = x + 1;
  }
  const o = (h - 2) * (tessellation + 1);
  for (let x = 0; x < tessellation; x++) {
    const i = o + x;
    indices[ii++] = n + x; indices[ii++] = i; indices[ii++] = i + 1;
  }
  for (let y = 1; y < h - 1; y++) {
    const o0 = (y - 1) * (tessellation + 1);
    const o1 =  y      * (tessellation + 1);
    for (let x = 0; x < tessellation; x++) {
      const i00 = o0 + x + 1, i01 = o1 + x + 1, i10 = o0 + x, i11 = o1 + x;
      indices[ii++] = i10; indices[ii++] = i00; indices[ii++] = i01;
      indices[ii++] = i10; indices[ii++] = i01; indices[ii++] = i11;
    }
  }
  return { positions, normals, uvs, indices };
}

// ---- Cylinder: unit cylinder, radius 1, axis along +Z, [0..1] high ----

export function buildCylinder(tessellation: number): GeometryData {
  const vertexCount = 4 * (tessellation + 1) + 2;
  const indexCount  = tessellation * 12;
  const positions = new Float32Array(vertexCount * 3);
  const normals   = new Float32Array(vertexCount * 3);
  // UV: side wraps as (angle/2π, z). Cap verts use planar projection
  // of (cos, sin) into [0,1]² centred at (0.5, 0.5).
  const uvs = new Float32Array(vertexCount * 2);
  const step = TWO_PI / tessellation;
  let oi = 0;
  let phi = 0;
  for (let i = 0; i <= tessellation; i++) {
    const c = Math.cos(phi), s = Math.sin(phi);
    const p0: [number, number, number] = [c, s, 0];
    const p1: [number, number, number] = [c, s, 1];
    const u = i / tessellation;
    const capU = 0.5 + 0.5 * c, capV = 0.5 + 0.5 * s;
    pushV3(positions, oi*3, p0); pushV3(normals, oi*3, [0,0,-1]);
    uvs[oi*2] = capU; uvs[oi*2 + 1] = capV;     oi++;   // bottom-cap vertex
    pushV3(positions, oi*3, p0); pushV3(normals, oi*3, [c, s, 0]);
    uvs[oi*2] = u;    uvs[oi*2 + 1] = 0;        oi++;   // side-bottom vertex
    pushV3(positions, oi*3, p1); pushV3(normals, oi*3, [0,0, 1]);
    uvs[oi*2] = capU; uvs[oi*2 + 1] = capV;     oi++;   // top-cap vertex
    pushV3(positions, oi*3, p1); pushV3(normals, oi*3, [c, s, 0]);
    uvs[oi*2] = u;    uvs[oi*2 + 1] = 1;        oi++;   // side-top vertex
    phi += step;
  }
  const bottom = oi;
  pushV3(positions, oi*3, [0,0,0]); pushV3(normals, oi*3, [0,0,-1]);
  uvs[oi*2] = 0.5; uvs[oi*2 + 1] = 0.5; oi++;
  const top = oi;
  pushV3(positions, oi*3, [0,0,1]); pushV3(normals, oi*3, [0,0, 1]);
  uvs[oi*2] = 0.5; uvs[oi*2 + 1] = 0.5; oi++;

  const indices = new Uint32Array(indexCount);
  let ii = 0;
  for (let i = 0; i < tessellation; i++) {
    const idx = (k: number, side: boolean, t: boolean): number =>
      4 * k + (t ? 2 : 0) + (side ? 1 : 0);
    const i00 = idx(i,     true, false), i10 = idx(i + 1, true, false);
    const i01 = idx(i,     true, true),  i11 = idx(i + 1, true, true);
    indices[ii++] = i00; indices[ii++] = i10; indices[ii++] = i01;
    indices[ii++] = i01; indices[ii++] = i10; indices[ii++] = i11;
    const j00 = idx(i, false, false), j10 = idx(i + 1, false, false);
    const j01 = idx(i, false, true),  j11 = idx(i + 1, false, true);
    indices[ii++] = j00; indices[ii++] = bottom; indices[ii++] = j10;
    indices[ii++] = j01; indices[ii++] = j11;     indices[ii++] = top;
  }
  return { positions, normals, uvs, indices };
}
