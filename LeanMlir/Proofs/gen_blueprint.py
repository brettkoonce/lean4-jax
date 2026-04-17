#!/usr/bin/env python3
"""Generate a dependency DAG of the VJP proof suite.

For each theorem in Proofs/, runs `#print axioms` via Lean, parses the
output, and emits a graphviz .dot file (+ optional .svg if `dot` is on
PATH) showing which axioms each theorem depends on.

The result visualizes "trust reduction": main theorems at the top,
framework primitives / opaque forwards / bundled VJPs (the 30 axioms)
at the bottom, edges showing "uses". Clustered by source file, colored
by axiom vs theorem.

Usage:
    python3 gen_blueprint.py

Output:
    blueprint.dot    (always)
    blueprint.svg    (if graphviz `dot` is available)

Render manually with:
    dot -Tsvg blueprint.dot -o blueprint.svg
    # or
    dot -Tpng blueprint.dot -o blueprint.png
"""
import subprocess
import re
import shutil
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# Config: every theorem/axiom we want to visualize, grouped by file.
# ════════════════════════════════════════════════════════════════

THEOREMS_BY_FILE = {
    "Tensor.lean": [
        # Primitives (axioms)
        "Proofs.pdiv", "Proofs.pdiv_comp", "Proofs.pdiv_add", "Proofs.pdiv_mul",
        "Proofs.pdiv_id", "Proofs.pdiv_const", "Proofs.pdiv_reindex",
        "Proofs.pdivMat_rowIndep",
        # Framework theorems
        "Proofs.pdiv_finset_sum",
        "Proofs.vjp_comp", "Proofs.biPath_has_vjp", "Proofs.elemwiseProduct_has_vjp",
        "Proofs.identity_has_vjp",
        "Proofs.vjpMat_comp", "Proofs.biPathMat_has_vjp", "Proofs.identityMat_has_vjp",
        "Proofs.pdivMat_scalarScale", "Proofs.pdivMat_transpose",
        "Proofs.pdivMat_matmul_left_const", "Proofs.pdivMat_matmul_right_const",
        "Proofs.matmul_left_const_has_vjp", "Proofs.matmul_right_const_has_vjp",
        "Proofs.scalarScale_has_vjp", "Proofs.transpose_has_vjp",
        "Proofs.rowwise_has_vjp_mat",
        "Proofs.pdiv3_comp", "Proofs.vjp3_comp", "Proofs.biPath3_has_vjp",
    ],
    "MLP.lean": [
        "Proofs.pdiv_dense", "Proofs.pdiv_dense_W",
        "Proofs.pdiv_relu", "Proofs.softmaxCE_grad",
        "Proofs.dense_has_vjp",
        "Proofs.dense_weight_grad_correct", "Proofs.dense_bias_grad_correct",
        "Proofs.relu_has_vjp",
    ],
    "CNN.lean": [
        "Proofs.conv2d", "Proofs.conv2d_has_vjp3",
        "Proofs.conv2d_weight_grad_has_vjp", "Proofs.conv2d_bias_grad_has_vjp",
        "Proofs.maxPool2", "Proofs.maxPool2_has_vjp3",
        "Proofs.conv2d_input_grad", "Proofs.conv2d_weight_grad", "Proofs.conv2d_bias_grad",
    ],
    "BatchNorm.lean": [
        "Proofs.pdiv_bnAffine", "Proofs.pdiv_bnCentered", "Proofs.pdiv_bnIstdBroadcast",
        "Proofs.bn_has_vjp", "Proofs.bnNormalize_has_vjp", "Proofs.bnAffine_has_vjp",
    ],
    "Residual.lean": [
        "Proofs.residual_has_vjp",
    ],
    "Depthwise.lean": [
        "Proofs.depthwiseConv2d", "Proofs.depthwise_has_vjp3",
        "Proofs.depthwise_weight_grad_has_vjp3", "Proofs.depthwise_bias_grad_has_vjp",
    ],
    "SE.lean": [
        "Proofs.seBlock_has_vjp",
    ],
    "LayerNorm.lean": [
        "Proofs.geluScalar", "Proofs.geluScalarDeriv", "Proofs.pdiv_gelu",
        "Proofs.layerNorm_has_vjp", "Proofs.gelu_has_vjp",
    ],
    "Attention.lean": [
        "Proofs.pdiv_softmax", "Proofs.mhsa_has_vjp_mat",
        "Proofs.softmax_has_vjp", "Proofs.rowSoftmax_has_vjp_mat",
        "Proofs.sdpa_back_Q_correct", "Proofs.sdpa_back_K_correct", "Proofs.sdpa_back_V_correct",
        "Proofs.layerNorm_per_token_has_vjp_mat",
        "Proofs.dense_per_token_has_vjp_mat", "Proofs.gelu_per_token_has_vjp_mat",
        "Proofs.transformerMlp_has_vjp_mat",
        "Proofs.transformerAttnSublayer_has_vjp_mat", "Proofs.transformerMlpSublayer_has_vjp_mat",
        "Proofs.transformerBlock_has_vjp_mat",
        "Proofs.transformerTower_has_vjp_mat",
        "Proofs.vit_body_has_vjp_mat",
    ],
}

# The 30 axioms across the suite (mirror README).
KNOWN_AXIOMS = {
    # Tensor.lean (8)
    "Proofs.pdiv", "Proofs.pdiv_comp", "Proofs.pdiv_add", "Proofs.pdiv_mul",
    "Proofs.pdiv_id", "Proofs.pdiv_const", "Proofs.pdiv_reindex", "Proofs.pdivMat_rowIndep",
    # MLP.lean (4)
    "Proofs.pdiv_dense", "Proofs.pdiv_dense_W", "Proofs.pdiv_relu", "Proofs.softmaxCE_grad",
    # CNN.lean (6)
    "Proofs.conv2d", "Proofs.conv2d_has_vjp3",
    "Proofs.conv2d_weight_grad_has_vjp", "Proofs.conv2d_bias_grad_has_vjp",
    "Proofs.maxPool2", "Proofs.maxPool2_has_vjp3",
    # BatchNorm.lean (3)
    "Proofs.pdiv_bnAffine", "Proofs.pdiv_bnCentered", "Proofs.pdiv_bnIstdBroadcast",
    # Depthwise.lean (4)
    "Proofs.depthwiseConv2d", "Proofs.depthwise_has_vjp3",
    "Proofs.depthwise_weight_grad_has_vjp3", "Proofs.depthwise_bias_grad_has_vjp",
    # LayerNorm.lean (3)
    "Proofs.geluScalar", "Proofs.geluScalarDeriv", "Proofs.pdiv_gelu",
    # Attention.lean (2)
    "Proofs.pdiv_softmax", "Proofs.mhsa_has_vjp_mat",
}

# Lean core axioms — present in every nontrivial term, noise for our graph.
CORE_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

FILE_COLORS = {
    "Tensor.lean":    "#e8f0ff",  # light blue - foundation
    "MLP.lean":       "#fff0e8",  # light orange
    "CNN.lean":       "#e8ffe8",  # light green
    "BatchNorm.lean": "#ffe8e8",  # light red
    "Residual.lean":  "#f0e8ff",  # light purple
    "Depthwise.lean": "#ffffe8",  # light yellow
    "SE.lean":        "#e8ffff",  # light cyan
    "LayerNorm.lean": "#ffe8ff",  # light magenta
    "Attention.lean": "#fff8e8",  # light amber - capstone
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DOT = Path(__file__).with_name("blueprint.dot")
OUT_SVG = Path(__file__).with_name("blueprint.svg")

# ════════════════════════════════════════════════════════════════
# Step 1: generate a Lean stub that #prints every theorem's axioms.
# ════════════════════════════════════════════════════════════════

def make_lean_stub(all_theorems):
    imports = [
        "LeanMlir.Proofs.Tensor",
        "LeanMlir.Proofs.MLP",
        "LeanMlir.Proofs.CNN",
        "LeanMlir.Proofs.BatchNorm",
        "LeanMlir.Proofs.Residual",
        "LeanMlir.Proofs.Depthwise",
        "LeanMlir.Proofs.SE",
        "LeanMlir.Proofs.LayerNorm",
        "LeanMlir.Proofs.Attention",
    ]
    lines = [f"import {imp}" for imp in imports]
    lines.append("")
    for t in all_theorems:
        lines.append(f"#print axioms {t}")
    return "\n".join(lines) + "\n"

def run_lean(stub_content):
    stub_path = Path("/tmp/blueprint_stub.lean")
    stub_path.write_text(stub_content)
    print(f"Running `lake env lean` on {len(stub_content.splitlines())} lines of stub...")
    proc = subprocess.run(
        ["lake", "env", "lean", str(stub_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    # Lean `#print` writes to stdout. Diagnostics ("depends on axioms") may come via stderr.
    return proc.stdout + "\n" + proc.stderr

# ════════════════════════════════════════════════════════════════
# Step 2: parse the "depends on axioms" output.
# ════════════════════════════════════════════════════════════════

DEP_RX = re.compile(r"'([^']+)' depends on axioms: \[([^\]]+)\]", re.DOTALL)

def parse_deps(text):
    deps = {}
    for m in DEP_RX.finditer(text):
        name = m.group(1)
        deps_list = [d.strip() for d in m.group(2).split(",")]
        deps[name] = set(deps_list)
    return deps

# ════════════════════════════════════════════════════════════════
# Step 3: emit graphviz DOT.
# ════════════════════════════════════════════════════════════════

def emit_dot(deps_map, groups):
    nodes_we_show = set()
    name_to_file = {}
    for file, names in groups.items():
        for n in names:
            nodes_we_show.add(n)
            name_to_file[n] = file

    lines = []
    lines.append("digraph blueprint {")
    lines.append('  rankdir=BT;')
    lines.append('  graph [fontname="Helvetica", label="VJP proof dependency graph\\n(axioms at bottom, composed theorems at top)", labelloc=t, fontsize=14];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];')
    lines.append('  edge [color="#999999", arrowsize=0.5, penwidth=0.8];')
    lines.append("")

    # Clusters per file, with color
    for file, names in groups.items():
        cluster_id = file.replace(".", "_").replace("-", "_")
        bg = FILE_COLORS.get(file, "#f0f0f0")
        lines.append(f'  subgraph cluster_{cluster_id} {{')
        lines.append(f'    label="{file}";')
        lines.append(f'    style="filled,rounded"; fillcolor="{bg}";')
        lines.append(f'    fontname="Helvetica-Bold"; fontsize=11;')
        for n in names:
            short = n.replace("Proofs.", "")
            is_axiom = n in KNOWN_AXIOMS
            # Axioms: red border, theorems: green border
            if is_axiom:
                border = '#c00000'; fill = '#ffe0e0'
                shape_extra = 'peripheries=2'
            else:
                border = '#007000'; fill = '#ffffff'
                shape_extra = ''
            attrs = f'label="{short}", color="{border}", fillcolor="{fill}"'
            if shape_extra:
                attrs += f', {shape_extra}'
            lines.append(f'    "{n}" [{attrs}];')
        lines.append("  }")
        lines.append("")

    # Edges: axiom/theorem dep → theorem (both in nodes_we_show)
    edge_count = 0
    for theorem, deps in deps_map.items():
        if theorem not in nodes_we_show:
            continue
        for dep in deps:
            if dep in CORE_AXIOMS:
                continue
            if dep == theorem:
                continue
            if dep in nodes_we_show:
                lines.append(f'  "{dep}" -> "{theorem}";')
                edge_count += 1
    lines.append("")
    lines.append("}")
    print(f"Emitted {edge_count} edges.")
    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════
# Main.
# ════════════════════════════════════════════════════════════════

def main():
    all_theorems = []
    seen = set()
    for file, names in THEOREMS_BY_FILE.items():
        for n in names:
            if n not in seen:
                all_theorems.append(n)
                seen.add(n)
    print(f"Visualizing {len(all_theorems)} theorems across {len(THEOREMS_BY_FILE)} files.")

    stub = make_lean_stub(all_theorems)
    raw = run_lean(stub)
    deps = parse_deps(raw)
    print(f"Parsed {len(deps)} dependency entries from Lean output.")

    missing = [t for t in all_theorems if t not in deps]
    if missing:
        print(f"WARNING: no deps found for {len(missing)} theorems:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    dot = emit_dot(deps, THEOREMS_BY_FILE)
    OUT_DOT.write_text(dot)
    print(f"Wrote: {OUT_DOT}")

    if shutil.which("dot"):
        print("Rendering SVG via graphviz...")
        subprocess.run(
            ["dot", "-Tsvg", str(OUT_DOT), "-o", str(OUT_SVG)],
            check=True,
        )
        print(f"Wrote: {OUT_SVG}")
    else:
        print("graphviz `dot` not on PATH — SVG not rendered.")
        print("Install with: sudo apt-get install graphviz")
        print(f"Then run: dot -Tsvg {OUT_DOT} -o {OUT_SVG}")

if __name__ == "__main__":
    main()
