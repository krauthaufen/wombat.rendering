// Bitfield encoding of a PipelineStateDescriptor into a single bigint.
//
// The encoded value is BOTH the pipeline-cache key AND the per-RO modeKey
// the partition kernel reads. There is no hash function — the encoding
// is exact and reversible. Two distinct descriptors always produce
// distinct keys; identical descriptors always produce identical keys.
//
// v1 layout supports up to MAX_ATTACHMENTS color attachments. Per-
// attachment data is packed sequentially after a fixed-width header.
// Stencil is encoded as a single "enabled" bit in v1; when enabled,
// the full stencil state is rejected (returned as an "open" key) — the
// design defers per-RO stencil derivation to v2.

import {
  type PipelineStateDescriptor,
  type AttachmentBlend,
  type BlendComponent,
  type CullMode,
  type FrontFace,
  type Topology,
} from "./descriptor.js";

const TOPOLOGY: readonly Topology[] = [
  "point-list",
  "line-list",
  "line-strip",
  "triangle-list",
  "triangle-strip",
];

const STRIP_INDEX_FORMAT: readonly (GPUIndexFormat | undefined)[] = [
  undefined,
  "uint16",
  "uint32",
];

const COMPARE: readonly GPUCompareFunction[] = [
  "never", "less", "equal", "less-equal",
  "greater", "not-equal", "greater-equal", "always",
];

const BLEND_FACTOR: readonly GPUBlendFactor[] = [
  "zero", "one",
  "src", "one-minus-src",
  "src-alpha", "one-minus-src-alpha",
  "dst", "one-minus-dst",
  "dst-alpha", "one-minus-dst-alpha",
  "src-alpha-saturated",
  "constant", "one-minus-constant",
];

const BLEND_OPERATION: readonly GPUBlendOperation[] = [
  "add", "subtract", "reverse-subtract", "min", "max",
];

const CULL_MODE: readonly CullMode[]  = ["none", "front", "back"];
const FRONT_FACE: readonly FrontFace[] = ["ccw", "cw"];

function indexOf<T>(arr: readonly T[], v: T, name: string): number {
  const i = arr.indexOf(v);
  if (i < 0) throw new Error(`pipelineCache: unknown ${name} "${String(v)}"`);
  return i;
}

// ─── Bit layout ────────────────────────────────────────────────────────
//
// Header (28 bits):
//   [ 2: 0]  topology         (3 bits, indexes TOPOLOGY)
//   [ 4: 3]  stripIndexFormat (2 bits, indexes STRIP_INDEX_FORMAT)
//   [ 5: 5]  frontFace        (1 bit)
//   [ 7: 6]  cullMode         (2 bits)
//   [ 8: 8]  depth enabled
//   [ 9: 9]  depth write
//   [12:10]  depth compare    (3 bits)
//   [13:13]  depth clamp (unclippedDepth)
//   [14:14]  stencil enabled  (full stencil state in side-bits region —
//                              currently unimplemented; throws if set)
//   [15:15]  alphaToCoverage
//   [19:16]  attachment count (4 bits, 0..MAX_ATTACHMENTS)
//   [27:20]  reserved
//
// Attachment slot (27 bits each), packed starting at bit 28:
//   [ 0: 0]  blend enabled
//   [ 3: 1]  color operation  (3 bits)
//   [ 7: 4]  color srcFactor  (4 bits)
//   [11: 8]  color dstFactor  (4 bits)
//   [14:12]  alpha operation  (3 bits)
//   [18:15]  alpha srcFactor  (4 bits)
//   [22:19]  alpha dstFactor  (4 bits)
//   [26:23]  writeMask        (4 bits, RGBA)
//
// At MAX_ATTACHMENTS=2 (v1 cap): header 28 + 2×27 = 82 bits → fits a
// bigint comfortably; CPU map keys take it natively.

export const MAX_ATTACHMENTS = 2;

const HEADER_BITS      = 28n;
const ATTACHMENT_BITS  = 27n;

// Field shifts (header)
const SH_TOPOLOGY      = 0n;
const SH_STRIP         = 3n;
const SH_FRONTFACE     = 5n;
const SH_CULLMODE      = 6n;
const SH_DEPTH_EN      = 8n;
const SH_DEPTH_WRITE   = 9n;
const SH_DEPTH_CMP     = 10n;
const SH_DEPTH_CLAMP   = 13n;
const SH_STENCIL_EN    = 14n;
const SH_ATC           = 15n;
const SH_ATT_COUNT     = 16n;

// Field shifts (attachment slot, relative to slot base)
const A_EN        = 0n;
const A_COLOR_OP  = 1n;
const A_COLOR_SRC = 4n;
const A_COLOR_DST = 8n;
const A_ALPHA_OP  = 12n;
const A_ALPHA_SRC = 15n;
const A_ALPHA_DST = 19n;
const A_MASK      = 23n;

function bits(value: number | bigint, width: bigint): bigint {
  const mask = (1n << width) - 1n;
  return (BigInt(value) & mask);
}

function encodeAttachment(att: AttachmentBlend): bigint {
  let s = 0n;
  s |= bits(att.enabled ? 1 : 0, 1n) << A_EN;
  // When disabled, color/alpha components are stored as zero so that
  // logically-equal descriptors encode identically. (descriptor.ts'
  // attachmentEquals also ignores color/alpha when !enabled.)
  if (att.enabled) {
    s |= bits(indexOf(BLEND_OPERATION, att.color.operation, "color blend op"), 3n) << A_COLOR_OP;
    s |= bits(indexOf(BLEND_FACTOR,    att.color.srcFactor, "color srcFactor"),  4n) << A_COLOR_SRC;
    s |= bits(indexOf(BLEND_FACTOR,    att.color.dstFactor, "color dstFactor"),  4n) << A_COLOR_DST;
    s |= bits(indexOf(BLEND_OPERATION, att.alpha.operation, "alpha blend op"), 3n) << A_ALPHA_OP;
    s |= bits(indexOf(BLEND_FACTOR,    att.alpha.srcFactor, "alpha srcFactor"),  4n) << A_ALPHA_SRC;
    s |= bits(indexOf(BLEND_FACTOR,    att.alpha.dstFactor, "alpha dstFactor"),  4n) << A_ALPHA_DST;
  }
  s |= bits(att.writeMask & 0xF, 4n) << A_MASK;
  return s;
}

function decodeAttachment(slot: bigint): AttachmentBlend {
  const en = ((slot >> A_EN) & 1n) === 1n;
  const writeMask = Number((slot >> A_MASK) & 0xFn);
  if (!en) {
    return { enabled: false, color: { srcFactor: "one", dstFactor: "zero", operation: "add" },
             alpha: { srcFactor: "one", dstFactor: "zero", operation: "add" }, writeMask };
  }
  const color: BlendComponent = {
    operation: BLEND_OPERATION[Number((slot >> A_COLOR_OP)  & 7n)]!,
    srcFactor: BLEND_FACTOR   [Number((slot >> A_COLOR_SRC) & 0xFn)]!,
    dstFactor: BLEND_FACTOR   [Number((slot >> A_COLOR_DST) & 0xFn)]!,
  };
  const alpha: BlendComponent = {
    operation: BLEND_OPERATION[Number((slot >> A_ALPHA_OP)  & 7n)]!,
    srcFactor: BLEND_FACTOR   [Number((slot >> A_ALPHA_SRC) & 0xFn)]!,
    dstFactor: BLEND_FACTOR   [Number((slot >> A_ALPHA_DST) & 0xFn)]!,
  };
  return { enabled: true, color, alpha, writeMask };
}

export function encodeModeKey(d: PipelineStateDescriptor): bigint {
  if (d.attachments.length > MAX_ATTACHMENTS) {
    throw new Error(
      `pipelineCache: ${d.attachments.length} attachments exceeds v1 cap of ${MAX_ATTACHMENTS}`,
    );
  }
  if (d.stencil !== undefined) {
    throw new Error(
      "pipelineCache: per-RO stencil state is not yet supported in v1 (use a static stencil at scene level)",
    );
  }
  let key = 0n;
  key |= bits(indexOf(TOPOLOGY, d.topology, "topology"), 3n) << SH_TOPOLOGY;
  key |= bits(indexOf(STRIP_INDEX_FORMAT, d.stripIndexFormat, "stripIndexFormat"), 2n) << SH_STRIP;
  key |= bits(indexOf(FRONT_FACE, d.frontFace, "frontFace"), 1n) << SH_FRONTFACE;
  key |= bits(indexOf(CULL_MODE,  d.cullMode,  "cullMode"),  2n) << SH_CULLMODE;
  if (d.depth !== undefined) {
    key |= 1n << SH_DEPTH_EN;
    if (d.depth.write) key |= 1n << SH_DEPTH_WRITE;
    key |= bits(indexOf(COMPARE, d.depth.compare, "depth compare"), 3n) << SH_DEPTH_CMP;
    if (d.depth.clamp) key |= 1n << SH_DEPTH_CLAMP;
  }
  if (d.alphaToCoverage) key |= 1n << SH_ATC;
  key |= bits(d.attachments.length, 4n) << SH_ATT_COUNT;
  for (let i = 0; i < d.attachments.length; i++) {
    const slot = encodeAttachment(d.attachments[i]!);
    key |= slot << (HEADER_BITS + BigInt(i) * ATTACHMENT_BITS);
  }
  return key;
}

export function decodeModeKey(key: bigint): PipelineStateDescriptor {
  const topology  = TOPOLOGY[Number((key >> SH_TOPOLOGY)  & 7n)]!;
  const strip     = STRIP_INDEX_FORMAT[Number((key >> SH_STRIP) & 3n)];
  const frontFace = FRONT_FACE[Number((key >> SH_FRONTFACE) & 1n)]!;
  const cullMode  = CULL_MODE [Number((key >> SH_CULLMODE)  & 3n)]!;
  const depthEn   = ((key >> SH_DEPTH_EN) & 1n) === 1n;
  const atc       = ((key >> SH_ATC)      & 1n) === 1n;
  const attCount  = Number((key >> SH_ATT_COUNT) & 0xFn);
  const slotMask  = (1n << ATTACHMENT_BITS) - 1n;
  const attachments: AttachmentBlend[] = [];
  for (let i = 0; i < attCount; i++) {
    const slot = (key >> (HEADER_BITS + BigInt(i) * ATTACHMENT_BITS)) & slotMask;
    attachments.push(decodeAttachment(slot));
  }
  const out: PipelineStateDescriptor = {
    topology,
    stripIndexFormat: strip,
    frontFace,
    cullMode,
    ...(depthEn ? { depth: {
      write:   ((key >> SH_DEPTH_WRITE) & 1n) === 1n,
      compare: COMPARE[Number((key >> SH_DEPTH_CMP) & 7n)]!,
      clamp:   ((key >> SH_DEPTH_CLAMP) & 1n) === 1n,
    } } : {}),
    attachments,
    alphaToCoverage: atc,
  };
  return out;
}
