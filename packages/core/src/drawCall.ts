// DrawCall — what to actually draw, once the pipeline + bindings
// are bound. Indexed and non-indexed variants; both can be
// instanced.

export interface IndexedDrawCall {
  readonly kind: "indexed";
  readonly indexCount: number;
  readonly instanceCount: number;
  readonly firstIndex: number;
  readonly baseVertex: number;
  readonly firstInstance: number;
}

export interface NonIndexedDrawCall {
  readonly kind: "non-indexed";
  readonly vertexCount: number;
  readonly instanceCount: number;
  readonly firstVertex: number;
  readonly firstInstance: number;
}

export type DrawCall = IndexedDrawCall | NonIndexedDrawCall;
