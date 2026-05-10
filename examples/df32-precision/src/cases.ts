// Test-case generators. Each case produces an array of (A, B) f64 pairs.

const seedRng = (seed: number) => {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) / 0x100000000);
  };
};

const rand = (rng: () => number, lo: number, hi: number) => lo + (hi - lo) * rng();

export type Pair = readonly [Float64Array, Float64Array];
export type Case = { name: string; description: string; pairs: Pair[] };

export const IDENTITY: Float64Array = (() => {
  const m = new Float64Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
})();

function randomMat(rng: () => number, scale: number): Float64Array {
  const m = new Float64Array(16);
  for (let i = 0; i < 16; i++) m[i] = rand(rng, -scale, scale);
  return m;
}

function translation(tx: number, ty: number, tz: number): Float64Array {
  const m = new Float64Array(16);
  m[0]  = 1;
  m[5]  = 1;
  m[10] = 1;
  m[15] = 1;
  m[3]  = tx;
  m[7]  = ty;
  m[11] = tz;
  return m;
}

function rotationAroundY(theta: number): Float64Array {
  const c = Math.cos(theta), s = Math.sin(theta);
  const m = new Float64Array(16);
  m[0] = c;  m[2] = s;
  m[5] = 1;
  m[8] = -s; m[10] = c;
  m[15] = 1;
  return m;
}

function mulRowMajor(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let col = 0; col < 4; col++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k]! * b[k * 4 + col]!;
      c[r * 4 + col] = s;
    }
  }
  return c;
}

// 1) Sanity: identity × random small. Result should match input exactly.
function caseSanity(): Case {
  const rng = seedRng(0xa110ce);
  const pairs: Pair[] = [];
  for (let i = 0; i < 32; i++) pairs.push([IDENTITY, randomMat(rng, 1)]);
  return {
    name: "sanity",
    description: "Identity × random — output must equal input.",
    pairs,
  };
}

// 2) Small magnitude: entries in [-1, 1]. Cancellation-free baseline.
function caseSmall(): Case {
  const rng = seedRng(0x5ba11);
  const pairs: Pair[] = [];
  for (let i = 0; i < 64; i++) pairs.push([randomMat(rng, 1), randomMat(rng, 1)]);
  return {
    name: "small",
    description: "Random matrices in [-1, 1] — baseline, no cancellation.",
    pairs,
  };
}

// 3) Geodetic: ECEF-scale translations canceled at varying δ.
//    Model = R₁ · T(+R_earth + δ),  View = T(-R_earth) · R₂
//    The ModelView product cancels the R_earth translation; only δ
//    survives. As δ → 0 the relative error of f32 explodes; df32 should
//    track f64.
function caseGeodetic(): Case {
  const rng = seedRng(0xea27e);
  const pairs: Pair[] = [];
  const R = 6.378137e6; // earth radius, m
  for (const delta of [1e3, 1, 1e-3, 1e-6]) {
    for (let i = 0; i < 8; i++) {
      const r1 = rotationAroundY(rand(rng, 0, Math.PI * 2));
      const r2 = rotationAroundY(rand(rng, 0, Math.PI * 2));
      const tForward  = translation(R + delta, rand(rng, -10, 10), rand(rng, -10, 10));
      const tBackward = translation(-R, 0, 0);
      const Model = mulRowMajor(r1, tForward);
      const View  = mulRowMajor(tBackward, r2);
      pairs.push([Model, View]);
    }
  }
  return {
    name: "geodetic",
    description: "ECEF-scale offsets canceling at δ ∈ {1e3, 1, 1e-3, 1e-6} m.",
    pairs,
  };
}

// 4) Composed rotations: chain of N random rotations, then multiply.
//    Tests error growth in a non-cancellation regime.
function caseComposed(): Case {
  const rng = seedRng(0xc0490);
  const pairs: Pair[] = [];
  for (let i = 0; i < 16; i++) {
    let A = rotationAroundY(rand(rng, 0, Math.PI * 2));
    let B = rotationAroundY(rand(rng, 0, Math.PI * 2));
    for (let k = 0; k < 50; k++) {
      A = mulRowMajor(A, rotationAroundY(rand(rng, 0, 0.1)));
      B = mulRowMajor(B, rotationAroundY(rand(rng, 0, 0.1)));
    }
    pairs.push([A, B]);
  }
  return {
    name: "composed",
    description: "50-deep rotation chains × random rotation.",
    pairs,
  };
}

export function allCases(): Case[] {
  return [caseSanity(), caseSmall(), caseGeodetic(), caseComposed()];
}
