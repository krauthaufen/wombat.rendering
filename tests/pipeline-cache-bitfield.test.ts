import { describe, test, expect } from "vitest";
import {
  encodeModeKey,
  decodeModeKey,
  descriptorEquals,
  DEFAULT_DESCRIPTOR,
  DEFAULT_ATTACHMENT_BLEND,
  type PipelineStateDescriptor,
} from "@aardworx/wombat.rendering/runtime";

const variants: PipelineStateDescriptor[] = [
  DEFAULT_DESCRIPTOR,

  // cullMode permutations
  { ...DEFAULT_DESCRIPTOR, cullMode: "none" },
  { ...DEFAULT_DESCRIPTOR, cullMode: "front" },

  // topology + strip index format
  { ...DEFAULT_DESCRIPTOR, topology: "triangle-strip", stripIndexFormat: "uint16" },
  { ...DEFAULT_DESCRIPTOR, topology: "line-list" },
  { ...DEFAULT_DESCRIPTOR, topology: "point-list" },

  // frontFace
  { ...DEFAULT_DESCRIPTOR, frontFace: "cw" },

  // depth permutations
  { ...DEFAULT_DESCRIPTOR, depth: { write: false, compare: "always", clamp: false } },
  { ...DEFAULT_DESCRIPTOR, depth: { write: true,  compare: "greater-equal", clamp: true } },
  // depth disabled
  { ...DEFAULT_DESCRIPTOR, depth: undefined as unknown as PipelineStateDescriptor["depth"] },

  // alphaToCoverage
  { ...DEFAULT_DESCRIPTOR, alphaToCoverage: true },

  // attachment blend on/off
  {
    ...DEFAULT_DESCRIPTOR,
    attachments: [{
      enabled: true,
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    }],
  },
  // premul over
  {
    ...DEFAULT_DESCRIPTOR,
    attachments: [{
      enabled: true,
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      writeMask: 0xF,
    }],
  },
  // additive
  {
    ...DEFAULT_DESCRIPTOR,
    attachments: [{
      enabled: true,
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
      writeMask: 0xF,
    }],
  },
  // writeMask color-only
  {
    ...DEFAULT_DESCRIPTOR,
    attachments: [{ ...DEFAULT_ATTACHMENT_BLEND, writeMask: 0x7 }],
  },
  // zero attachments
  { ...DEFAULT_DESCRIPTOR, attachments: [] },
  // two attachments
  {
    ...DEFAULT_DESCRIPTOR,
    attachments: [
      DEFAULT_ATTACHMENT_BLEND,
      { enabled: true,
        color: { srcFactor: "src", dstFactor: "one-minus-src", operation: "max" },
        alpha: { srcFactor: "one", dstFactor: "zero",          operation: "min" },
        writeMask: 0xF },
    ],
  },
];

describe("pipelineCache/bitfield", () => {
  test("encode is deterministic", () => {
    for (const d of variants) {
      expect(encodeModeKey(d)).toBe(encodeModeKey(d));
    }
  });

  test("distinct descriptors produce distinct keys", () => {
    const seen = new Map<bigint, PipelineStateDescriptor>();
    for (const d of variants) {
      const k = encodeModeKey(d);
      const prev = seen.get(k);
      if (prev !== undefined) {
        // Distinct variants must never collide.
        expect(descriptorEquals(prev, d), `collision: ${JSON.stringify(prev)} vs ${JSON.stringify(d)}`).toBe(true);
      }
      seen.set(k, d);
    }
  });

  test("round-trip preserves structural equality", () => {
    for (const d of variants) {
      const k = encodeModeKey(d);
      const back = decodeModeKey(k);
      expect(descriptorEquals(d, back)).toBe(true);
    }
  });

  test("equivalent descriptors with disabled blend encode identically", () => {
    const a: PipelineStateDescriptor = {
      ...DEFAULT_DESCRIPTOR,
      attachments: [{
        enabled: false,
        // Different color/alpha than DEFAULT_ATTACHMENT_BLEND but blend
        // is disabled, so these fields are semantically dead.
        color: { srcFactor: "dst", dstFactor: "one", operation: "max" },
        alpha: { srcFactor: "dst", dstFactor: "one", operation: "max" },
        writeMask: 0xF,
      }],
    };
    const b: PipelineStateDescriptor = {
      ...DEFAULT_DESCRIPTOR,
      attachments: [DEFAULT_ATTACHMENT_BLEND],
    };
    expect(encodeModeKey(a)).toBe(encodeModeKey(b));
    expect(descriptorEquals(a, b)).toBe(true);
  });

  test("rejects out-of-range attachments", () => {
    const tooMany: PipelineStateDescriptor = {
      ...DEFAULT_DESCRIPTOR,
      attachments: [DEFAULT_ATTACHMENT_BLEND, DEFAULT_ATTACHMENT_BLEND, DEFAULT_ATTACHMENT_BLEND],
    };
    expect(() => encodeModeKey(tooMany)).toThrow(/attachments exceeds/);
  });

  test("rejects stencil (v1 not supported)", () => {
    const withStencil: PipelineStateDescriptor = {
      ...DEFAULT_DESCRIPTOR,
      stencil: {
        readMask: 0xFF, writeMask: 0xFF,
        front: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        back:  { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
      },
    };
    expect(() => encodeModeKey(withStencil)).toThrow(/stencil/);
  });
});
