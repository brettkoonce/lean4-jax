import LeanMlir

/-! # AlphaGo — Bestiary entry

AlphaGo (Silver et al., Nature 2016 --- "Mastering the game of Go
with deep neural networks and tree search") is the system that beat
Lee Sedol 4--1 in March 2016 --- the first time a computer defeated
a top-ranked human Go professional. It also made ``neural network +
Monte Carlo Tree Search'' the reference template for board-game AI
for half a decade.

We already have \texttt{AlphaZero.lean} in the bestiary covering
AlphaGo Zero + AlphaZero (the 2017--2018 successors). Those
architectures are both one two-headed network: a shared convolutional
body feeding parallel policy + value heads. AlphaGo the original
was \emph{three separate networks}:

1. **Policy network** ($p_\sigma$, then $p_\rho$). A 13-layer conv
   net trained first on $\sim$30M human-expert Go positions
   (supervised $p_\sigma$) then further refined via self-play RL
   ($p_\rho$). Predicts probability of each move given the board.
   $\sim$4.5M params.
2. **Value network** ($v_\theta$). A 13-layer conv net followed by
   fully-connected heads, predicting win probability from the
   current position. $\sim$5M params.
3. **Rollout policy** ($p_\pi$). A \emph{shallow} linear softmax
   over $\sim$$10^5$ hand-crafted binary features (3$\times$3
   stone patterns, liberties, responses to previous move, etc.).
   Used inside MCTS for fast tree playouts --- the full 13-layer
   net would be too slow for the thousands of simulations per move.

Plus hand-crafted \textbf{48 input feature planes} instead of raw
board state. Many of those planes are parity-specific indicator
features (``stone captured last move,'' ``ladder works,'' liberty
counts). Two years later, AlphaGo Zero threw all of this out ---
just the raw 19$\times$19 board history + one two-headed network
trained purely from self-play --- and \emph{played better}. That
simplification is the lesson.

## Architecture differences vs AlphaGo Zero

| Aspect               | AlphaGo (2016)        | AlphaGo Zero (2017)      |
|----------------------|-----------------------|--------------------------|
| Input features       | 48 hand-crafted planes | 17 raw-history planes   |
| Networks             | 3 separate            | 1 two-headed            |
| Training data        | Human games + RL      | Self-play only          |
| Rollout inside MCTS  | Fast shallow net      | Value head directly     |
| Parameter count      | $\sim$10M total       | $\sim$25M (bigger body) |
| Elo (full system)    | 3140                  | 5185                    |

The lesson ran counter to the field's working assumption in 2016 ---
``of course you need hand-crafted features and multiple networks for
a game this complex.'' AlphaGo Zero showed that neither was
necessary. The raw-board, shared-body template then generalized
trivially to chess and shogi as AlphaZero.

## Variants

- `alphaGoPolicyNet` --- 13-layer conv policy net (19$\times$19$\times$48
  input, 362 output moves)
- `alphaGoValueNet`  --- 13-layer conv value net with FC head (scalar
  win-probability output)
- `alphaGoRollout`   --- the shallow linear softmax used inside MCTS
- `tinyAlphaGo`      --- compact fixture
-/

-- ════════════════════════════════════════════════════════════════
-- § Policy network (13 conv layers at 192 filters)
-- ════════════════════════════════════════════════════════════════
-- Input: 19×19 × 48 (hand-crafted feature planes)
-- Output: 19×19 probability map + 1 "pass" move = 362 logits

def alphaGoPolicyNet : NetSpec where
  name := "AlphaGo policy network"
  imageH := 19
  imageW := 19
  layers := [
    -- First conv: 5×5 filter on 48-ch input (the paper's design)
    .convBn 48 192 5 1 .same,
    -- 11 more 3×3 convs at 192 channels
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    -- Final 1×1 conv: 192 → 1 (per-position move logit)
    -- The 19×19 output + pass move is handled by the forward pass
    -- reshaping the 19×19×1 output + concatenating a learned scalar.
    .conv2d 192 1 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Value network (13 conv layers + FC head → scalar win-probability)
-- ════════════════════════════════════════════════════════════════

def alphaGoValueNet : NetSpec where
  name := "AlphaGo value network"
  imageH := 19
  imageW := 19
  layers := [
    -- Same 13-layer conv body as the policy net
    .convBn 48 192 5 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    .convBn 192 192 3 1 .same,
    -- Head: 1×1 conv to 1 channel, flatten, then FC 361 → 256 → 1
    .conv2d 192 1 1 .same .relu,
    .flatten,
    .dense 361 256 .relu,
    .dense 256 1 .identity    -- scalar win-probability (tanh in real)
  ]

-- ════════════════════════════════════════════════════════════════
-- § Rollout policy (shallow softmax over hand-crafted features)
-- ════════════════════════════════════════════════════════════════
-- In the real AlphaGo the rollout policy is a linear softmax over
-- ~10^5 binary hand-crafted features. No convs, no hidden layers —
-- just a sparse linear model with ~1M params by weight count, but
-- almost all features zero at any given board so the forward pass
-- is ~microseconds. Fast enough for MCTS rollouts at thousands of
-- simulations per move.

def alphaGoRollout : NetSpec where
  name := "AlphaGo rollout policy (hand-crafted linear softmax)"
  imageH := 19       -- board resolution
  imageW := 19
  layers := [
    -- Stand-in: flatten the 19×19 board + dense to 362 moves. Real
    -- AlphaGo's rollout has ~1M params as a feature-hashed sparse
    -- linear model; our dense stand-in is about right for shape.
    .flatten,
    .dense 361 362 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyAlphaGo — compact 9×9 fixture
-- ════════════════════════════════════════════════════════════════

def tinyAlphaGo : NetSpec where
  name := "tiny-AlphaGo policy (9×9 board)"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 12 64 5 1 .same,     -- 12 input planes, 64 filters
    .convBn 64 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .conv2d 64 1 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — AlphaGo"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The 2016 Lee Sedol match. Three networks (policy + value +"
  IO.println "  rollout), 48 hand-crafted feature planes, MCTS outer loop."

  summarize alphaGoPolicyNet
  summarize alphaGoValueNet
  summarize alphaGoRollout
  summarize tinyAlphaGo

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Policy and value networks are"
  IO.println "    plain convBn stacks; rollout is a flat dense projection."
  IO.println "  • AlphaGo Zero (2017) replaced all three networks with a"
  IO.println "    single two-headed net (see AlphaZero.lean) trained purely"
  IO.println "    via self-play on the raw board. Went from 3140 Elo to"
  IO.println "    5185 Elo by simplifying, not complicating."
  IO.println "  • The 48 hand-crafted feature planes were the crown jewel of"
  IO.println "    Go engineering circa 2015. Each encoded a specific Go"
  IO.println "    concept: liberties, captures, 'ladder works', 3x3 stone"
  IO.println "    patterns, etc. AlphaGo Zero dropped them entirely. If"
  IO.println "    there's a one-sentence lesson from the AlphaGo lineage:"
  IO.println "    'features the network can learn on its own, it will'."
  IO.println "  • The rollout network is there for a reason not visible in"
  IO.println "    the architecture: MCTS runs thousands of simulations per"
  IO.println "    move, and each simulation plays dozens of moves through"
  IO.println "    the position. Running the full 13-layer net that many"
  IO.println "    times per move was too slow in 2016, so they trained a"
  IO.println "    fast shallow model whose only job was to produce reasonable"
  IO.println "    move priors fast enough for the tree search."
  IO.println "  • See AlphaZero.lean for AlphaGo Zero + AlphaZero; see"
  IO.println "    MuZero.lean for the fully model-based descendant that"
  IO.println "    learns the game dynamics itself."
