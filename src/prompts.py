INTUITIVE_PROMPT = """
You will be given a small number of training examples for an ARC-like puzzle.
Each example is a pair of grids: an input grid and the corresponding output grid.

Write clear, *general* instructions that describe the transformation from input to output.
Focus on: color semantics, connected components, symmetries, counting, copying/mirroring, cropping, tiling, etc.
Avoid overfitting to exact coordinates unless truly invariant.

Return ONLY the instruction text (no JSON).
""".strip()

FOLLOW_PROMPT = """
You are given:
- A set of natural-language instructions that describe a grid transformation.
- A single input grid (2D array of integers 0..9).

Apply the instructions to the input grid and return JSON ONLY in the format:
{"grid": [[...], [...], ...]}

Do not add commentary, markdown, or extra keys.
Be strict: the grid must be the correct size and contain only integers 0..9.
""".strip()

REVISION_PROMPT = """
Your previous instructions were applied to training examples and produced incorrect outputs.
You will see expected outputs and your produced outputs (actual).
Revise the instructions to FIX the specific errors while retaining what works and keeping generality.

Return ONLY the revised instruction text (no commentary).
""".strip()

POOLING_PROMPT = """
Below are several partially-correct instruction sets.
Synthesize ONE unified, concise instruction that:
1) Preserves parts that consistently worked,
2) Fixes recurring mistakes,
3) Generalizes across all training examples.

Return ONLY the final instruction text (no commentary).
""".strip()
