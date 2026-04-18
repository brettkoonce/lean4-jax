import LeanMlir

/-! # Whisper — Bestiary entry

Whisper (Radford et al., OpenAI, 2022 — "Robust Speech Recognition
via Large-Scale Weak Supervision") is the reference audio model for
the 2020s: an encoder-decoder transformer trained on 680,000 hours
of multilingual / multitask weak supervision scraped from the web.
It does transcription, translation, language identification, and
voice activity detection with a single checkpoint --- task selection
is just a special token in the decoder input.

The architectural story is the shortest in the bestiary:

**Encoder-decoder transformer over log-mel spectrograms.**

That's it. Same transformer blocks you've seen everywhere else. The
interesting work is in the data collection, the multitask
token-prefix interface, and the sheer scale --- none of which lives
in the NetSpec.

## Anatomy

```
  30-second audio clip
         │
         ▼
  log-mel spectrogram (80 mel bins × 3000 time frames)
         │
         ▼  2-layer 1D conv stem (kernel 3, 2nd stride 2)
  audio tokens: (1500, D)
         │
         ▼  + sinusoidal positional embeddings
    Encoder : N × transformer blocks
         │
         ▼
    audio features (1500, D) --------┐
                                     │
    BOS + special tokens             │
         │                           │
         ▼  + learned position emb   │  cross-attention
    Decoder : N × transformer blocks ◄──────────┘
         │
         ▼  (LM head, tied to text-embedding)
  next-token logits over 51865 BPE tokens
```

Both encoder and decoder share the same depth / width / head count
per variant. The decoder's cross-attention at every block is how
audio features leak into the text side.

## Special tokens: multitask via prefix

The decoder input isn't just text --- it's a prefix of special
tokens describing what task to do. Example:

```
  <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> hello world <|endoftext|>
```

Swap `<|en|>` for `<|es|>` to pick Spanish. Swap `<|transcribe|>` for
`<|translate|>` to translate to English instead of transcribing.
That's the whole multitask interface --- no architectural changes.

## Variants (paper, all encoder-decoder, same depth both sides)

- `whisperTiny`   — 4  layers, dim 384,  heads 6,  mlp 1536 → $\sim$39M
- `whisperBase`   — 6  layers, dim 512,  heads 8,  mlp 2048 → $\sim$74M
- `whisperSmall`  — 12 layers, dim 768,  heads 12, mlp 3072 → $\sim$244M
- `whisperMedium` — 24 layers, dim 1024, heads 16, mlp 4096 → $\sim$769M
- `whisperLarge`  — 32 layers, dim 1280, heads 20, mlp 5120 → $\sim$1.55B
  (large-v1/v2/v3 share this architecture; differences are training-data only)
- `whisperSmallDecoder` — illustrative decoder spec (small config; others scale)
- `tinyWhisper` — fixture

## NetSpec simplifications

- The 2-layer 1D conv stem (~1M params at small, ~2.5M at large) is
  omitted --- its contribution is under 1\% of total params, and our
  primitives are 2D-conv-flavored.
- Shown as two NetSpecs per variant: one encoder, one decoder. Real
  Whisper runs the encoder once per audio clip and the decoder
  autoregressively per output token; that's orchestration, not
  architecture.
- Decoder text-embedding is a \texttt{.dense vocab $\to$ D}
  stand-in, same trick BERT / GPT use. Output projection is tied
  to the embedding, so no extra head needed.
-/

-- ════════════════════════════════════════════════════════════════
-- § Whisper encoder variants (audio side)
-- ════════════════════════════════════════════════════════════════

def whisperTiny : NetSpec where
  name := "Whisper tiny encoder"
  imageH := 1500      -- post-stem audio token count (3000 frames / 2)
  imageW := 1
  layers := [
    .transformerEncoder 384 6 1536 4
  ]

def whisperBase : NetSpec where
  name := "Whisper base encoder"
  imageH := 1500
  imageW := 1
  layers := [
    .transformerEncoder 512 8 2048 6
  ]

def whisperSmall : NetSpec where
  name := "Whisper small encoder"
  imageH := 1500
  imageW := 1
  layers := [
    .transformerEncoder 768 12 3072 12
  ]

def whisperMedium : NetSpec where
  name := "Whisper medium encoder"
  imageH := 1500
  imageW := 1
  layers := [
    .transformerEncoder 1024 16 4096 24
  ]

def whisperLarge : NetSpec where
  name := "Whisper large encoder (v1/v2/v3)"
  imageH := 1500
  imageW := 1
  layers := [
    .transformerEncoder 1280 20 5120 32
  ]

-- ════════════════════════════════════════════════════════════════
-- § Whisper decoder (illustrated at small size; others scale)
-- ════════════════════════════════════════════════════════════════
-- The decoder reads text-token inputs, cross-attends to the encoder's
-- audio features, and predicts the next BPE token. Self-attn is causal
-- (upper-triangular mask — training detail, not a parameter). nQueries
-- is 0 because there are no DETR-style learned object queries; text
-- tokens come from the embedding below.

def whisperSmallDecoder : NetSpec where
  name := "Whisper small decoder (shared shape)"
  imageH := 448        -- max decoded text tokens per clip
  imageW := 1
  layers := [
    -- Text token embedding + tied LM head (51865 BPE tokens, same trick
    -- as GPT/BERT: one .dense stands for both the input embedding and
    -- the output projection because they share weights).
    .dense 51865 768 .identity,
    -- 12 decoder blocks (self-attn + cross-attn + FFN). nQueries = 0.
    .transformerDecoder 768 12 3072 12 0
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyWhisper — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinyWhisper : NetSpec where
  name := "tiny-Whisper encoder"
  imageH := 256
  imageW := 1
  layers := [
    .transformerEncoder 128 2 512 2
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  context     : {spec.imageH} tokens"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Whisper"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Encoder-decoder transformer over log-mel spectrograms."
  IO.println "  Multitask via token-prefix; 680,000 hours of weak supervision."

  summarize whisperTiny
  summarize whisperBase
  summarize whisperSmall
  summarize whisperMedium
  summarize whisperLarge
  summarize whisperSmallDecoder
  summarize tinyWhisper

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Encoder is .transformerEncoder,"
  IO.println "    decoder is .transformerDecoder (from DETR) with nQueries=0"
  IO.println "    — same primitive that encodes any seq2seq decoder with"
  IO.println "    self-attn + cross-attn + FFN blocks."
  IO.println "  • Whisper's encoder and decoder are the same size per variant"
  IO.println "    (same layers, dim, heads). For Whisper small: encoder 85M,"
  IO.println "    decoder 113M, text-embedding 40M → 238M total vs paper 244M."
  IO.println "    Same ~2-3% agreement at every size. The 2-layer conv1d stem"
  IO.println "    (~1M at small) is omitted for NetSpec simplicity."
  IO.println "  • Whisper's multitask interface is the interesting design"
  IO.println "    move: swap a language token and the SAME network does a"
  IO.println "    different language. Swap a task token and it translates"
  IO.println "    instead of transcribing. No fine-tuning, no architecture"
  IO.println "    change — all the multitasking is a prompt-engineering"
  IO.println "    trick at the decoder input."
  IO.println "  • large-v1, large-v2, large-v3 are architecturally identical."
  IO.println "    Deltas: v2 trained 2.5x longer, v3 used 128 mel bins"
  IO.println "    instead of 80 (one stem-layer input size change, param"
  IO.println "    count impact ~0.1%)."
