// §7 derived-uniform recipe table — leaf-only flavour.
//
// Every recipe is a self-contained leaf: reads constituents directly,
// computes inline (one or two df32 muls + collapse), writes the final
// f32 mat4/mat3 to the main heap drawHeader. No intermediates, no
// layering, no dependents graph.
//
// Two consumers of the same df32 product (e.g. ModelViewTrafo and
// ModelViewProjTrafo for the same RO) recompute Model·View twice —
// trivial vs the encode-time savings of one flat dispatch.
//
// All df32 math runs through the verified primitives in
// uberKernel.wgsl.ts; the final consumer-facing values are f32.

/** Recipe IDs. Numeric values are stable — the kernel switches on them. */
export const enum RecipeId {
  /** Model.fwd → f32 mat4. */
  ModelTrafo            = 0,
  /** Model.bwd → f32 mat4. */
  ModelTrafoInv         = 1,
  /** (Model.bwd)ᵀ upper-3×3 → f32 mat3. */
  NormalMatrix          = 2,

  /** View.fwd · Model.fwd → f32 mat4. */
  ModelViewTrafo        = 3,
  /** Model.bwd · View.bwd → f32 mat4. */
  ModelViewTrafoInv     = 4,
  /** Proj.fwd · View.fwd · Model.fwd → f32 mat4. */
  ModelViewProjTrafo    = 5,
  /** Model.bwd · View.bwd · Proj.bwd → f32 mat4. */
  ModelViewProjTrafoInv = 6,

  /** View.fwd → f32 mat4. */
  ViewTrafo             = 7,
  /** View.bwd → f32 mat4. */
  ViewTrafoInv          = 8,
  /** Proj.fwd · View.fwd → f32 mat4. */
  ViewProjTrafo         = 9,
  /** View.bwd · Proj.bwd → f32 mat4. */
  ViewProjTrafoInv      = 10,
  /** Proj.fwd → f32 mat4. */
  ProjTrafo             = 11,
  /** Proj.bwd → f32 mat4. */
  ProjTrafoInv          = 12,
}

/** Number of constituent inputs each recipe consumes (1, 2, or 3). */
const INPUT_COUNT: Record<RecipeId, 1 | 2 | 3> = {
  [RecipeId.ModelTrafo]:            1,
  [RecipeId.ModelTrafoInv]:         1,
  [RecipeId.NormalMatrix]:          1,
  [RecipeId.ModelViewTrafo]:        2,
  [RecipeId.ModelViewTrafoInv]:     2,
  [RecipeId.ModelViewProjTrafo]:    3,
  [RecipeId.ModelViewProjTrafoInv]: 3,
  [RecipeId.ViewTrafo]:             1,
  [RecipeId.ViewTrafoInv]:          1,
  [RecipeId.ViewProjTrafo]:         2,
  [RecipeId.ViewProjTrafoInv]:      2,
  [RecipeId.ProjTrafo]:             1,
  [RecipeId.ProjTrafoInv]:          1,
};

export function recipeInputCount(id: RecipeId): 1 | 2 | 3 {
  return INPUT_COUNT[id];
}

/** Output shape — drives drawHeader byte allocation. */
export type RecipeOutput = "mat4" | "mat3";

const OUTPUT: Record<RecipeId, RecipeOutput> = {
  [RecipeId.ModelTrafo]:            "mat4",
  [RecipeId.ModelTrafoInv]:         "mat4",
  [RecipeId.NormalMatrix]:          "mat3",
  [RecipeId.ModelViewTrafo]:        "mat4",
  [RecipeId.ModelViewTrafoInv]:     "mat4",
  [RecipeId.ModelViewProjTrafo]:    "mat4",
  [RecipeId.ModelViewProjTrafoInv]: "mat4",
  [RecipeId.ViewTrafo]:             "mat4",
  [RecipeId.ViewTrafoInv]:          "mat4",
  [RecipeId.ViewProjTrafo]:         "mat4",
  [RecipeId.ViewProjTrafoInv]:      "mat4",
  [RecipeId.ProjTrafo]:             "mat4",
  [RecipeId.ProjTrafoInv]:          "mat4",
};

export function recipeOutput(id: RecipeId): RecipeOutput {
  return OUTPUT[id];
}

/** Which side (forward/backward) of which Trafo3d each input slot draws from. */
export type ConstituentRef =
  | "Model.fwd" | "Model.bwd"
  | "View.fwd"  | "View.bwd"
  | "Proj.fwd"  | "Proj.bwd";

const INPUTS: Record<RecipeId, readonly ConstituentRef[]> = {
  [RecipeId.ModelTrafo]:            ["Model.fwd"],
  [RecipeId.ModelTrafoInv]:         ["Model.bwd"],
  [RecipeId.NormalMatrix]:          ["Model.bwd"],
  // df_mul args are ordered (outer, inner) so the math product
  // matches Aardvark's `View.mul(Model).forward = View.fwd · Model.fwd`.
  [RecipeId.ModelViewTrafo]:        ["View.fwd",  "Model.fwd"],
  [RecipeId.ModelViewTrafoInv]:     ["Model.bwd", "View.bwd"],
  [RecipeId.ModelViewProjTrafo]:    ["Proj.fwd",  "View.fwd", "Model.fwd"],
  [RecipeId.ModelViewProjTrafoInv]: ["Model.bwd", "View.bwd", "Proj.bwd"],
  [RecipeId.ViewTrafo]:             ["View.fwd"],
  [RecipeId.ViewTrafoInv]:          ["View.bwd"],
  [RecipeId.ViewProjTrafo]:         ["Proj.fwd",  "View.fwd"],
  [RecipeId.ViewProjTrafoInv]:      ["View.bwd",  "Proj.bwd"],
  [RecipeId.ProjTrafo]:             ["Proj.fwd"],
  [RecipeId.ProjTrafoInv]:          ["Proj.bwd"],
};

export function recipeInputs(id: RecipeId): readonly ConstituentRef[] {
  return INPUTS[id];
}

/** Public uniform name (one per recipe). */
const NAME: Record<RecipeId, string> = {
  [RecipeId.ModelTrafo]:            "ModelTrafo",
  [RecipeId.ModelTrafoInv]:         "ModelTrafoInv",
  [RecipeId.NormalMatrix]:          "NormalMatrix",
  [RecipeId.ModelViewTrafo]:        "ModelViewTrafo",
  [RecipeId.ModelViewTrafoInv]:     "ModelViewTrafoInv",
  [RecipeId.ModelViewProjTrafo]:    "ModelViewProjTrafo",
  [RecipeId.ModelViewProjTrafoInv]: "ModelViewProjTrafoInv",
  [RecipeId.ViewTrafo]:             "ViewTrafo",
  [RecipeId.ViewTrafoInv]:          "ViewTrafoInv",
  [RecipeId.ViewProjTrafo]:         "ViewProjTrafo",
  [RecipeId.ViewProjTrafoInv]:      "ViewProjTrafoInv",
  [RecipeId.ProjTrafo]:             "ProjTrafo",
  [RecipeId.ProjTrafoInv]:          "ProjTrafoInv",
};

/** All recipe IDs in switch order. */
export const ALL_RECIPES: readonly RecipeId[] = [
  RecipeId.ModelTrafo, RecipeId.ModelTrafoInv, RecipeId.NormalMatrix,
  RecipeId.ModelViewTrafo, RecipeId.ModelViewTrafoInv,
  RecipeId.ModelViewProjTrafo, RecipeId.ModelViewProjTrafoInv,
  RecipeId.ViewTrafo, RecipeId.ViewTrafoInv,
  RecipeId.ViewProjTrafo, RecipeId.ViewProjTrafoInv,
  RecipeId.ProjTrafo, RecipeId.ProjTrafoInv,
];

const BY_NAME: ReadonlyMap<string, RecipeId> = new Map(
  ALL_RECIPES.map(id => [NAME[id], id] as const),
);

export const DERIVED_UNIFORM_NAMES: ReadonlySet<string> =
  new Set(ALL_RECIPES.map(id => NAME[id]));

export function recipeIdByName(name: string): RecipeId | undefined {
  return BY_NAME.get(name);
}

export function recipeName(id: RecipeId): string {
  return NAME[id];
}
