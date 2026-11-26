// The format an effect expects each of its textures in.
//
// An image has no colour space of its own: the same pixels are a colour
// (to be decoded from sRGB when sampled) under one shader and plain
// numbers (a normal, a roughness) under another. So it is the EFFECT that
// says what it expects, per texture:
//
//   const pbr = withTextureFormats(effect(vs, fs), {
//     BaseColorTex: "rgba8unorm-srgb",   // colours: decoded on sampling
//     NormalTex:    "rgba8unorm",        // numbers: taken as they are
//   });
//
// Whoever draws with the effect then keeps the texture in that format:
// the heap's atlas has a set of pages per format and places the texture on
// the declared one; the classic path allocates its texture with it. A
// texture supplied under two effects that disagree is simply kept twice.
// A binding with no declaration is kept in the format the texture value
// itself names (an image without one: "rgba8unorm").
//
// (This is a side table keyed by the effect object. The shader language
// has no syntax for it yet; when it has, the declarations can come from
// there and land in the same table.)

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { ITexture } from "./texture.js";

const declared = new WeakMap<object, ReadonlyMap<string, GPUTextureFormat>>();

/** Declare the formats `effect` expects its textures in (by the names the shader gives them). Returns the effect. */
export function withTextureFormats<E extends object>(effect: E, formats: Readonly<Record<string, GPUTextureFormat>>): E {
  const merged = new Map(declared.get(effect) ?? []);
  for (const [name, format] of Object.entries(formats)) merged.set(name, format);
  declared.set(effect, merged);
  return effect;
}

/** The format `effect` declared for a texture binding (the binding's name, with or without the `_view` a compiled shader appends). */
export function textureFormatOf(effect: object, binding: string): GPUTextureFormat | undefined {
  const formats = declared.get(effect);
  if (formats === undefined) return undefined;
  return formats.get(binding) ?? (binding.endsWith("_view") ? formats.get(binding.slice(0, -5)) : formats.get(`${binding}_view`));
}

/** Formats whose texels are the same bytes, read differently: one may stand in for the other without converting anything. */
const SAME_BYTES: readonly (readonly GPUTextureFormat[])[] = [
  ["rgba8unorm", "rgba8unorm-srgb"], ["bgra8unorm", "bgra8unorm-srgb"],
];
const reinterpretable = (a: GPUTextureFormat, b: GPUTextureFormat): boolean =>
  a === b || SAME_BYTES.some((family) => family.includes(a) && family.includes(b));

function inFormat(t: ITexture, format: GPUTextureFormat): ITexture {
  if (t.kind !== "host") return t;
  const src = t.source;
  const own = src.format ?? "rgba8unorm";
  if (own === format || !reinterpretable(own, format)) return t;
  return { kind: "host", source: { ...src, format } };
}

const views = new WeakMap<aval<ITexture>, Map<GPUTextureFormat, aval<ITexture>>>();

/**
 * `texture`, kept in `format`: the same pixels under another reading (a
 * texture value that already has that format, or whose bytes cannot be
 * read that way, comes back as it is). One aval per (texture, format), so
 * every draw asking for the same reading shares it, and its atlas entry.
 */
export function textureInFormat(texture: aval<ITexture>, format: GPUTextureFormat | undefined): aval<ITexture> {
  if (format === undefined) return texture;
  let byFormat = views.get(texture);
  if (byFormat === undefined) { byFormat = new Map(); views.set(texture, byFormat); }
  let view = byFormat.get(format);
  if (view === undefined) {
    if (texture.isConstant) {
      const value = texture.force(/* allow-force */);
      const read = inFormat(value, format);
      view = read === value ? texture : AVal.constant(read);
    } else {
      view = texture.map((t) => inFormat(t, format));
    }
    byFormat.set(format, view);
  }
  return view;
}
