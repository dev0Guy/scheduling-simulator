"""Generate the SVG diagrams used by docs/experiment_log.md.

Run: .venv/bin/python scripts/docs_tools/make_diagrams.py
Writes docs/diagrams/*.svg (one per model M0..M10, plus environment, lineage, results).
Green boxes = what this model adds over the model it was built from.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = """
<style>
  text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #1f2328; }
  .t { font-size: 13px; font-weight: 600; }
  .s { font-size: 11.5px; fill: #57606a; }
  .h { font-size: 17px; font-weight: 700; }
  .base { fill: #eaf1fb; stroke: #5b8fd6; stroke-width: 1.4; }
  .new  { fill: #e3f6e8; stroke: #2da44e; stroke-width: 2.2; }
  .side { fill: #fff6e0; stroke: #d4a72c; stroke-width: 1.4; }
  .io   { fill: #f6f8fa; stroke: #8c959f; stroke-width: 1.2; }
  .bad  { fill: #ffebe9; stroke: #cf222e; stroke-width: 1.6; }
  .arr  { stroke: #6e7781; stroke-width: 1.6; fill: none; marker-end: url(#a); }
  .sarr { stroke: #d4a72c; stroke-width: 1.6; fill: none; marker-end: url(#a2); stroke-dasharray: 5 4; }
</style>
<defs>
  <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e7781"/></marker>
  <marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#d4a72c"/></marker>
</defs>
"""


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'{STYLE}<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


def wrap(text, width):
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return lines + ([cur] if cur else [])


def box(x, y, w, h, title, sub, cls, wrap_at=78):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" class="{cls}"/>'
    if isinstance(sub, (list, tuple)):
        lines = [ln for item in sub for ln in wrap(item, wrap_at)]
    else:
        lines = wrap(sub, wrap_at) if sub else []
    total = 16 + 15 * len(lines)
    ty = y + (h - total) / 2 + 13
    out += f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" class="t">{escape(title)}</text>'
    for i, line in enumerate(lines):
        out += f'<text x="{x + w / 2}" y="{ty + 17 + 15 * i}" text-anchor="middle" class="s">{escape(line)}</text>'
    return out


def legend(x, y):
    items = [("new", "added in this model"), ("base", "carried over"), ("side", "side input")]
    out = ""
    for i, (c, label) in enumerate(items):
        out += f'<rect x="{x + i * 190}" y="{y}" width="16" height="16" rx="3" class="{c}"/>'
        out += f'<text x="{x + i * 190 + 22}" y="{y + 12}" class="s">{label}</text>'
    return out


# ---------------------------------------------------------------- model diagrams
ENC20 = ("Job encoder", "20 -> 256 -> 256 (Linear + LayerNorm + ReLU, twice), shared across jobs", "base")
POOL = ("Attention pooling", "score each job (256 -> 32 -> 1, tanh), weighted sum, Linear -> global summary 256", "base")
SELF = ("Job self-attention (4 heads)", "every job attends to every other job, finished jobs masked; residual + LayerNorm", "new")
SCORER768 = ("Scorer, once per job", "[global | machine embedding | job] 768 -> 128 -> 128 -> 1;  skip head 256 -> 64 -> 1", "base")
OUT = ("11 action logits", "[skip, job 1 ... job 10]  ->  softmax  ->  action", "io")

FIT_SIDE = ("Side input", "job usage = max(usage)/255   |   machine room = min(capacity - usage)/255")


def fit_row(masked=False, fuse="concat"):
    sub = "Q from job usage, K and V from machine room (64-d); weight = sigmoid(Q.K / 8)"
    if masked:
        sub += "; weight = 0 where usage > room"
    return ("Fit attention" + (" with feasibility mask" if masked else ""), sub, "new")


FUSE_CONCAT = ("Concat + project", "[summary 256 | fit 64] -> Linear -> LayerNorm -> ReLU -> 256", "new")
FUSE_MUL = ("Multiply (per channel)", "summary x (1 + tanh(Linear(fit))), instead of concat + project", "new")

MODELS = {
    "M0": dict(
        title="M0: baseline (np7aenua / 6b195fqa). Cannot see job usage.",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival", "io", None), ENC20 + (None,), POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M1": dict(
        title="M1: M0 + queue_pressure + job self-attention (g231w4jg). Best on E1.",
        rows=[("Per-job input", "M0 features + queue_pressure (global and per job)", "new", None), ENC20 + (None,),
              SELF + (None,), POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M5": dict(
        title="M5: M0 + job usage everywhere (raw inputs). z4sayojp",
        rows=[("Per-job input (B x 10 x 21)", "M0 features + job usage = max(usage)/255", "new", None),
              ENC20[:1] + ("21 -> 256 -> 256, now sees usage", "base", None),
              POOL + (None,),
              ("Scorer, once per job", "[global | machine emb | job | job usage | machine room] 770 -> 128 -> 128 -> 1", "new",
               ("Side input", "raw machine room = min(capacity - usage)/255")),
              OUT + (None,)],
    ),
    "M6": dict(
        title="M6: usage kept out of the encoder, joined later by fit attention. iusicvre",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "io", None), ENC20 + (None,),
              fit_row() + (FIT_SIDE,), FUSE_CONCAT + (None,), POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M7": dict(
        title="M7: M6 + job self-attention. emt46quc",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "io", None), ENC20 + (None,),
              ("Fit attention", "Q from job usage, K and V from machine room (64-d); weight = sigmoid(Q.K / 8)", "base", FIT_SIDE),
              ("Concat + project", "[summary 256 | fit 64] -> Linear -> LayerNorm -> ReLU -> 256", "base", None),
              SELF + (None,), POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M8": dict(
        title="M8: M7 with multiplication instead of concat. bv1hk6lb (worse than M7)",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "io", None), ENC20 + (None,),
              ("Fit attention", "Q from job usage, K and V from machine room (64-d); weight = sigmoid(Q.K / 8)", "base", FIT_SIDE),
              FUSE_MUL + (None,),
              ("Job self-attention (4 heads)", "every job attends to every other job, finished jobs masked; residual + LayerNorm", "base", None),
              POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M9": dict(
        title="M9: M7 + feasibility mask in the fit attention. etudt4qo. Current best (+2.28% vs SJF).",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "io", None), ENC20 + (None,),
              fit_row(masked=True) + (FIT_SIDE,),
              ("Concat + project", "[summary 256 | fit 64] -> Linear -> LayerNorm -> ReLU -> 256", "base", None),
              ("Job self-attention (4 heads)", "every job attends to every other job, finished jobs masked; residual + LayerNorm", "base", None),
              POOL + (None,), SCORER768 + (None,), OUT + (None,)],
    ),
    "M11": dict(
        title="M11: M9 + not-created mask, on E1. j5ojvpto",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival; all 20 values set to 0 for jobs not yet arrived", "new",
               ("Not-created mask", "status = NOT_CREATED (job has not arrived): all values zeroed")),
              ENC20 + (None,),
              ("Fit attention with feasibility mask", "weight = 0 where usage > room; usage of unarrived jobs set to 0", "base", FIT_SIDE),
              ("Concat + project", "[summary 256 | fit 64] -> Linear -> LayerNorm -> ReLU -> 256", "base", None),
              ("Job self-attention (4 heads)", "keys masked: finished and not-yet-arrived jobs; residual + LayerNorm; unarrived outputs set to 0", "new", None),
              ("Attention pooling", "not-yet-arrived jobs get weight 0; weighted sum -> global summary 256", "new", None),
              SCORER768 + (None,), OUT + (None,)],
    ),
    "M10": dict(
        title="M10: M9 trained with entropy coefficient 0.05 -> 0.01. ne5qv1d2 (same result as M9)",
        rows=[("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "io", None), ENC20 + (None,),
              ("Fit attention with feasibility mask", "weight = 0 where usage > room", "base", FIT_SIDE),
              ("Concat + project", "[summary 256 | fit 64] -> 256", "base", None),
              ("Job self-attention (4 heads)", "finished jobs masked; residual + LayerNorm", "base", None),
              POOL + (None,), SCORER768 + (None,),
              ("Training change", "entropy coefficient annealed 0.05 -> 0.01 (M9 used 0.02 -> 0.002)", "new", None),
              OUT + (None,)],
    ),
}


def model_svg(key):
    spec = MODELS[key]
    rows = spec["rows"]
    W, X, BW, GAP = 900, 40, 520, 26
    def row_h(sub):
        return 58 if len(wrap(sub, 78)) <= 1 else 74
    H = 90 + sum(row_h(r[1]) + GAP for r in rows) + 50
    body = f'<text x="{X}" y="30" class="h">{escape(spec["title"])}</text>' + legend(X, 44)
    y = 80
    prev_bottom = None
    for title, sub, cls, side in rows:
        BH = row_h(sub)
        if prev_bottom is not None:
            body += f'<path d="M{X + BW / 2} {prev_bottom} L{X + BW / 2} {y}" class="arr"/>'
        body += box(X, y, BW, BH, title, sub, cls)
        if side:
            sx, sw = X + BW + 40, 290
            body += box(sx, y, sw, BH, side[0], side[1], "side", wrap_at=44)
            body += f'<path d="M{sx} {y + BH / 2} L{X + BW} {y + BH / 2}" class="sarr"/>'
        prev_bottom = y + BH
        y += BH + GAP
    body += f'<text x="{X}" y="{y + 4}" class="s">Critic (value) head, all models: global summary 256 -> 256 -> 256 -> 1.</text>'
    body += f'<text x="{X}" y="{y + 22}" class="s">PPO: lr 7e-6, 2048 steps, batch 64, 5 epochs, gamma = lambda = 0.995, 2M steps.</text>'
    return svg(W, H, body)


# ---------------------------------------------------------------- environment diagram
def env_svg():
    W, H = 900, 520
    b = '<text x="40" y="30" class="h">Environments: what a job looks like and what fits on the machine</text>'
    b += '<text x="40" y="52" class="s">1 machine (capacity 255), 10 jobs, 3 resources, 10-tick usage window, episode cap 110 steps. Job usage is one constant value on every resource and tick.</text>'
    scale = 560 / 255
    def bar(y, label, items, ok):
        o = f'<text x="40" y="{y - 8}" class="t">{escape(label)}</text>'
        o += f'<rect x="40" y="{y}" width="{255 * scale}" height="34" rx="6" class="io"/>'
        x = 40
        cols = {63: "#8fc1ff", 128: "#ffd27a", 255: "#ff9d9d"}
        for v in items:
            o += f'<rect x="{x}" y="{y}" width="{v * scale}" height="34" rx="6" fill="{cols[v]}" stroke="#57606a"/>'
            o += f'<text x="{x + v * scale / 2}" y="{y + 22}" text-anchor="middle" class="t">{v}</text>'
            x += v * scale
        o += f'<text x="{40 + max(sum(items), 255) * scale + 14}" y="{y + 22}" class="t" style="fill:{"#1a7f37" if ok else "#cf222e"}">{"fits" if ok else "does not fit"}</text>'
        return o
    b += '<text x="40" y="92" class="h">E1: every job uses 255</text>'
    b += bar(130, "One job fills the machine, so jobs run strictly one after another", [255], True)
    b += '<text x="40" y="190" class="h">E4: each job uses 63, 128 or 255 (random, 1/3 each)</text>'
    b += bar(238, "63 + 63 + 128 = 254", [63, 63, 128], True)
    b += bar(300, "63 + 128 = 191", [63, 128], True)
    b += bar(362, "128 + 128 = 256 > 255", [128, 128], False)
    b += bar(424, "63 + 255 = 318 > 255 (a 255 job needs the machine to itself)", [63, 255], False)
    b += '<text x="40" y="500" class="s">The optimal schedule on E4 can run several jobs at once, so a good policy has to check what fits, not only which job is shortest.</text>'
    return svg(W, H, b)


# ---------------------------------------------------------------- lineage + results
RUNS = [
    # id, model, margin vs SJF (%), label
    ("cf2zhwti", "M0", -164.6),
    ("z4sayojp", "M5", -45.81),
    ("iusicvre", "M6", -36.98),
    ("bv1hk6lb", "M8", -25.3),
    ("emt46quc", "M7", -16.87),
    ("ne5qv1d2", "M10", 2.31),
    ("etudt4qo", "M9", 2.28),
]


def results_svg():
    rows = [("Optimal (exact)", "script", 11.9)] + sorted([(m, rid, v) for rid, m, v in RUNS], key=lambda r: -r[2])
    W, H = 900, 100 + len(rows) * 46
    b = '<text x="40" y="30" class="h">E4 results: mean reward margin vs SJF = (PPO - SJF) / |SJF|</text>'
    b += '<text x="40" y="52" class="s">5000 deterministic episodes, same job sets as the SJF baseline rjzcixv2 (mean reward -50.43). Right of the line = better than SJF.</text>'
    x0, scale = 770, 3.0
    b += f'<line x1="{x0}" y1="72" x2="{x0}" y2="{80 + len(rows) * 46 - 8}" stroke="#1f2328" stroke-width="1.5"/>'
    b += f'<text x="{x0}" y="68" text-anchor="middle" class="s">0% (= SJF)</text>'
    for i, (name, rid, v) in enumerate(rows):
        y = 80 + i * 46
        c = "#5b8fd6" if name.startswith("Optimal") else ("#2da44e" if v > 0 else "#cf222e")
        x = x0 + min(v, 0) * scale
        b += f'<rect x="{x}" y="{y}" width="{abs(v) * scale}" height="28" rx="4" fill="{c}"/>'
        if v >= 0:
            b += f'<text x="{x0 + v * scale + 8}" y="{y + 19}" class="t">{v:+.2f}%</text>'
        else:
            b += f'<text x="{x0 + 8}" y="{y + 19}" class="t">{v:+.2f}%</text>'
        b += f'<text x="40" y="{y + 19}" class="t">{escape(name)}</text><text x="170" y="{y + 19}" class="s">{escape(rid)}</text>'
    return svg(W, H, b)


def lineage_svg():
    W, H = 900, 520
    b = '<text x="40" y="30" class="h">How the models were built from each other (E4 margin vs SJF)</text>'
    def node(x, y, title, sub, res, good=False, bad=False):
        cls = "new" if good else ("bad" if bad else "base")
        o = box(x, y, 190, 74, title, sub, cls, wrap_at=30)
        o += f'<text x="{x + 95}" y="{y + 66}" text-anchor="middle" class="t">{res}</text>'
        return o
    def arrow(x1, y1, x2, y2, label=""):
        o = f'<path d="M{x1} {y1} L{x2} {y2}" class="arr"/>'
        if label:
            o += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" text-anchor="middle" class="s">{escape(label)}</text>'
        return o
    b += node(40, 70, "M0", "cannot see usage", "-164.6%", bad=True)
    b += node(40, 200, "M5", "usage as raw inputs", "-45.8%")
    b += node(330, 70, "M6", "fit attention (concat)", "-37.0%")
    b += node(330, 200, "M7", "+ job self-attention", "-16.9%")
    b += node(620, 130, "M8", "multiply instead of concat", "-25.3% (worse)", bad=True)
    b += node(330, 340, "M9", "+ feasibility mask", "+2.28% (beats SJF)", good=True)
    b += node(620, 340, "M10", "M9 + more entropy", "+2.31% (same as M9)")
    b += arrow(135, 144, 135, 200, "add usage")
    b += arrow(230, 107, 330, 107, "")
    b += '<text x="280" y="98" text-anchor="middle" class="s">usage out of</text><text x="280" y="112" text-anchor="middle" class="s">the encoder</text>'
    b += arrow(425, 144, 425, 200, "")
    b += '<text x="468" y="176" class="s">+ self-attention</text>'
    b += arrow(520, 237, 620, 190, "")
    b += '<text x="570" y="200" class="s">multiply</text>'
    b += arrow(425, 274, 425, 340, "")
    b += '<text x="468" y="312" class="s">+ mask infeasible pairs</text>'
    b += arrow(520, 377, 620, 377, "entropy")
    b += '<text x="40" y="470" class="s">Baseline for all E4 runs: SJF rjzcixv2, mean reward -50.43. Optimal (exact search): -44.41 (+11.9%).</text>'
    b += '<text x="40" y="490" class="s">Green = beats SJF, red = worse than the model it came from or far from SJF, blue = intermediate step.</text>'
    return svg(W, H, b)


# ---------------------------------------------------------------- pfrw14vh (M9 + 3 additions + new training setup)
def pfrw_svg():
    W, X, BW = 1060, 40, 500
    SX, SW = 590, 430
    rows = [
        ("Per-job input (B x 10 x 20)", "status (16-d) + size, wait, ttl, arrival  (no usage)", "base"),
        ("Job encoder", "20 -> 256 -> 256 (Linear + LayerNorm + ReLU, twice), shared across jobs", "base"),
        ("Fit attention + feasibility mask, per resource", "Q from job usage (3 numbers), K and V from machine room (3 numbers); weight = 0 unless the job fits on EVERY resource", "new"),
        ("Concat + project", "[summary 256 | fit 64] -> Linear -> LayerNorm -> ReLU -> 256", "base"),
        ("Job self-attention (4 heads)", "every job attends to every other job, finished jobs masked; residual + LayerNorm", "base"),
        ("Attention pooling", "score each job (256 -> 32 -> 1, tanh), weighted sum -> global summary 256", "base"),
        ("Scorer, once per job", "[global 256 | machine 256 | job 256 | pair 64] = 832 -> 128 -> 128 -> 1; skip head 256 -> 64 -> 1", "new"),
        ("11 action logits", "[skip, job 1 ... job 10] -> softmax -> action", "io"),
    ]
    heights = [58, 58, 90, 58, 58, 58, 74, 58]
    GAP = 24
    ys, y = [], 90
    for h in heights:
        ys.append(y)
        y += h + GAP
    body = '<text x="40" y="30" class="h">pfrw14vh: M9 plus three additions and a new training setup</text>' + legend(X, 44)
    for i, (t, sub, cls) in enumerate(rows):
        if i:
            body += f'<path d="M{X + BW / 2} {ys[i - 1] + heights[i - 1]} L{X + BW / 2} {ys[i]}" class="arr"/>'
        body += box(X, ys[i], BW, heights[i], t, sub, cls, wrap_at=76)

    def side(y0, h, title, sub, cls="side"):
        return box(SX, y0, SW, h, title, sub, cls, wrap_at=56)

    # A: per-resource fit inputs -> fit attention row
    ya, ha = ys[2], heights[2]
    body += side(ya, ha, "1. Per-resource fit inputs",
                 "Old: one number per job (max usage) and one per machine (min room). New: one number per resource, 3 each, each taken over time only.", "new")
    body += f'<path d="M{SX} {ya + ha / 2} L{X + BW} {ya + ha / 2}" class="sarr"/>'
    # C: dynamic machine embedding (aligned with pooling row), arrow down to scorer
    yc, hc = ys[5] - 34, 112
    body += side(yc, hc, "2. Dynamic machine embedding",
                 "Free-capacity grid (3 resources x 10 ticks) / 255 -> shared row encoder 10 -> 64 -> 64 -> join the 3 rows (192) -> 256 -> 256. Added to the static machine embedding. Last layer starts at zero, so training starts like M9.", "new")
    # B: pair features aligned with scorer row
    yb, hb = ys[6] + 22, 128
    body += side(yb, hb, "3. Pair features (machine, job): 9 numbers",
                 "pending, fits now, delay until it fits / 10, finish time, headroom per resource (3), dominant share, packing. Delay checks every start offset 0 to 9 against the free grid. Linear 9 -> 64.", "new")
    ys_mid = ys[6] + heights[6] / 2
    body += f'<path d="M{SX} {yb + 30} L{X + BW + 28} {yb + 30} L{X + BW + 28} {ys_mid} L{X + BW} {ys_mid}" class="sarr"/>'
    body += f'<path d="M{SX} {yc + hc - 24} L{X + BW + 14} {yc + hc - 24} L{X + BW + 14} {ys_mid - 8} L{X + BW} {ys_mid - 8}" class="sarr"/>'

    # training changes
    yt = ys[-1] + heights[-1] + 36
    body += box(X, yt, 980, 96, "Training changes (not part of the network)",
                "8 envs x 512 steps per update (earlier runs: 4 x 2048). Learning rate 2e-5 (from example.py) decaying linearly to 5% (M9: constant 7e-6). target_kl 0.03 stops an epoch early when the policy moves too far. Entropy 0.05 -> 0.01 (M9: 0.02 -> 0.002). Evaluation uses 50-episode checks and the final test loads the best saved checkpoint, not the last model.", "new", wrap_at=150)
    H = yt + 96 + 50
    body += f'<text x="{X}" y="{H - 34}" class="s">Critic (value) head: global summary 256 -> 256 -> 256 -> 1.   PPO: batch 64, 5 epochs, gamma = lambda = 0.995, 2M steps.</text>'
    body += f'<text x="{X}" y="{H - 16}" class="s">Everything else is unchanged from M9. Run pfrw14vh (olive-fire-343).</text>'
    return svg(W, H, body)


# ---------------------------------------------------------------- full network flow (input -> action + value)
HEAD_DIM = 64   # size of one attention head in network_full_svg (embedding size / 4)


def network_full_svg(single=False):
    """single=True: the single-usage environment (E2): every job has one usage value, so the
    fit attention takes one number per job and one per machine instead of three."""
    W = 1100
    K = "1 number" if single else "3 numbers"
    UD = "B×10" if single else "B×10×3"
    RD = "B×1" if single else "B×1×3"
    P_EXT = "626,129" if single else "626,513"
    P_TOT = "898,644" if single else "899,028"
    title = ("The current network for the single-usage environment (E2): one usage number per job"
             if single else "The current network: from the observation to the action and the value")
    out = f'<text x="40" y="30" class="h">{title}</text>' + legend(40, 44)

    def bx(x, y, w, title, sub, cls="base"):
        wrap_at = max(20, int(w / 6.6))
        items = sub if isinstance(sub, (list, tuple)) else [sub]
        n = sum(len(wrap(i, wrap_at)) for i in items)
        h = 40 + 16 * n
        return box(x, y, w, h, title, sub, cls, wrap_at=wrap_at), h

    def arrow(x1, y1, x2, y2, cls="arr"):
        return f'<path d="M{x1} {y1} L{x2} {y2}" class="{cls}"/>'

    LX, LW = 40, 470       # main column
    RX, RW = 590, 470      # side column
    GAP = 26

    # observation
    s_, ho = bx(40, 70, 1020, "Observation",
                ["Output: machine usage and capacity B×1×3×10;  job usage B×10×3×10;  status, size, ttl, arrival, wait, start and finish times B×10;  time B×1"], "io")
    out += s_; obs_bottom = 70 + ho

    # ---- main column: positions first, drawn after
    boxes = []
    y = obs_bottom + 70
    def add(title, sub, cls="base"):
        nonlocal y
        s_, h = bx(LX, y, LW, title, sub, cls)
        boxes.append((y, h)); out_parts.append(s_); y += h + GAP
    out_parts = []
    add("Per-job features",
        ["In: status B×10, size, wait, ttl, arrival B×10 each", "Out: B×10×20", "status embedding (16 learned numbers) + the 4 scaled numbers"])
    add("Job encoder",
        ["In: B×10×20", "Out: B×10×256", "Linear 20→256, LayerNorm, ReLU, twice; the same weights for every job"])
    add("Fit attention + feasibility mask",
        [f"In: job usage {UD}, machine room {RD}", "Out: fit vector B×10×64",
         f"Query = what the job needs (its usage, {K} → 64)",
         f"Key = what the machine has free (its room, {K} → 64)",
         f"Value = the machine information that gets passed on (its room, {K} → 64)",
         ("weight = sigmoid(Query · Key ÷ 8), set to 0 unless the job fits (usage ≤ room); output = weight × Value" if single else "weight = sigmoid(Query · Key ÷ 8), set to 0 unless the job fits on every resource; output = weight × Value")], "base" if single else "new")
    add("Concat + project",
        ["In: job summary B×10×256 + fit vector B×10×64", "Out: B×10×256", "concatenate (320), Linear 320→256, LayerNorm, ReLU"])
    add("Job self-attention (4 heads)",
        ["In: B×10×256 (finished jobs masked out as keys)", "Out: B×10×256",
         "Query = what this job looks for in the other jobs",
         "Key = what each job offers to be compared on",
         "Value = the information a job passes on when another job attends to it",
         f"all three are linear projections of the same job summaries (4 heads of {HEAD_DIM}); then residual + LayerNorm"])
    add("Attention pooling",
        ["In: B×10×256", "Out: global summary B×256",
         "no query, key or value: a small network (256→32→1) scores how much each job matters, a softmax over the jobs gives weights, the summary is the weighted sum, then Linear 256→256"])
    main_end = y - GAP
    ys = [b[0] for b in boxes]; hs = [b[1] for b in boxes]
    for sp in out_parts:
        out += sp
    out += arrow(LX + LW / 2, obs_bottom, LX + LW / 2, ys[0])
    for i in range(len(ys) - 1):
        out += arrow(LX + LW / 2, ys[i] + hs[i], LX + LW / 2, ys[i + 1])
    y_fit, h_fit = ys[2], hs[2]
    y_sa, h_sa = ys[4], hs[4]

    # ---- side column
    sy = y_fit
    if single:
        s_, hr1 = bx(RX, sy, RW, "Fit inputs (one number each)",
                     ["In: job usage B×10×3×10, machine usage and capacity B×1×3×10",
                      "Out: job usage B×10 (max over resources and time ÷ 255), machine room B×1 (min over resources and time of capacity − usage ÷ 255)",
                      "in E2 a job uses the same value on every resource, so one number is enough"], "base")
    else:
        s_, hr1 = bx(RX, sy, RW, "Per-resource fit inputs",
                     ["In: job usage B×10×3×10, machine usage and capacity B×1×3×10",
                      "Out: job usage B×10×3 (max over time ÷ 255), machine room B×1×3 (min of capacity − usage ÷ 255)"], "new")
    out += s_
    out += arrow(RX + RW / 2, obs_bottom, RX + RW / 2, sy)
    out += arrow(RX, sy + hr1 / 2, LX + LW, y_fit + h_fit / 2, "sarr")
    dy = sy + hr1 + 22
    s_, hr2 = bx(RX, dy, RW, "Dynamic machine embedding",
                 ["In: free grid B×1×3×10 (capacity − usage, ÷ 255)", "Out: B×1×256",
                  "row encoder 10→64→64 for each resource, join the 3 rows (192), Linear 192→256, ReLU, Linear 256→256; the last layer starts at zero"], "new")
    out += s_
    py = dy + hr2 + 22
    s_, hr3 = bx(RX, py, RW, "Pair features (machine, job)",
                 ["In: free grid, job usage and size, status", "Out: B×1×10×9",
                  ("fits now, ticks until it fits, finish time, headroom per resource (3, identical in E2), dominant share, packing score, pending flag" if single else "fits now, ticks until it fits, finish time, headroom per resource (3), dominant share, packing score, pending flag")], "new")
    out += s_
    side_end = py + hr3

    # ---- feature vector
    fy = max(main_end, side_end) + 60
    s_, hf = bx(40, fy, 1020, "Feature vector",
                ["In: global summary 256, 10 job summaries 2560, pair features 90, dynamic machine embedding 256", "Out: B×3162   (the actor reads all of it, the critic only the first 256 numbers)"], "base")
    out += s_
    # frame around the shared feature extractor
    out += f'<rect x="26" y="{obs_bottom + 10}" width="1048" height="{fy - 14 - obs_bottom - 10 + 0}" rx="14" fill="none" stroke="#8c959f" stroke-width="1.4" stroke-dasharray="7 5"/>'
    out += f'<text x="300" y="{obs_bottom + 30}" class="s">Shared feature extractor ({P_EXT} parameters), used by the actor and the critic</text>'
    out += arrow(LX + LW / 2, main_end, LX + LW / 2, fy)
    out += f'<path d="M{LX + LW} {y_sa + h_sa / 2} L{LX + LW + 24} {y_sa + h_sa / 2} L{LX + LW + 24} {fy}" class="arr"/>'
    out += f'<text x="{LX + LW + 32}" y="{y_sa + h_sa / 2 - 6}" class="s">jobs</text>'
    out += f'<path d="M{RX} {dy + hr2 / 2} L{LX + LW + 76} {dy + hr2 / 2} L{LX + LW + 76} {fy}" class="arr"/>'
    out += arrow(RX + RW - 60, py + hr3, RX + RW - 60, fy)

    # ---- actor and critic
    ay = fy + hf + 44
    out += f'<text x="40" y="{ay - 14}" class="t">Actor head (140,674 parameters)</text>'
    out += f'<text x="880" y="{ay - 14}" class="t" text-anchor="end">Critic (131,841 parameters)</text>'
    AX, AW = 40, 640
    def abx(x, y, w, title, sub, cls="base"):
        s_, h = bx(x, y, w, title, sub, cls); return s_, h
    s_, ha1 = abx(AX, ay, AW, "Machine embedding and pair projection",
                  ["In: dynamic machine embedding B×1×256, pair features B×1×10×9", "Out: machine B×1×256, pair B×1×10×64",
                   "static machine embedding + the dynamic one; pair features Linear 9→64, ReLU"]); out += s_
    a2 = ay + ha1 + GAP
    s_, ha2 = abx(AX, a2, AW, "Scoring input, one row per job",
                  ["In: global B×256, machine B×1×256, job summaries B×10×256, pair B×10×64", "Out: B×10×832   (256 + 256 + 256 + 64 for each job)"]); out += s_
    a3 = a2 + ha2 + GAP
    s_, ha3 = abx(AX, a3, 310, "Score network", ["In: B×10×832", "Out: job scores B×10", "832→128→128→1, the same weights for every job"]); out += s_
    s_, ha3b = abx(AX + 330, a3, 310, "Skip head", ["In: global summary B×256", "Out: skip score B×1", "256→64→1"]); out += s_
    a4 = a3 + max(ha3, ha3b) + GAP
    s_, ha4 = abx(AX, a4, AW, "Action logits", ["In: skip score B×1, job scores B×10", "Out: B×11   [ skip, job 1 ... job 10 ]"], "io"); out += s_
    a5 = a4 + ha4 + GAP
    s_, ha5 = abx(AX, a5, AW, "Policy", ["In: logits B×11", "Out: one action per observation, 0 to 10", "softmax, then sample (training) or arg max (evaluation); 0 = skip, k = start job k − 1 on the machine"], "io"); out += s_
    out += arrow(AX + AW / 2, fy + hf, AX + AW / 2, ay)
    out += arrow(AX + AW / 2, ay + ha1, AX + AW / 2, a2)
    out += arrow(AX + 155, a2 + ha2, AX + 155, a3)
    out += arrow(AX + 485, a2 + ha2, AX + 485, a3)
    out += arrow(AX + 155, a3 + ha3, AX + 155, a4)
    out += arrow(AX + 485, a3 + ha3b, AX + 485, a4)
    out += arrow(AX + AW / 2, a4 + ha4, AX + AW / 2, a5)

    CX, CW = 730, 330
    s_, hc1 = abx(CX, ay, CW, "Global summary", ["In: feature vector B×3162", "Out: B×256 (its first 256 numbers)"]); out += s_
    c2 = ay + hc1 + GAP
    s_, hc2 = abx(CX, c2, CW, "Value network", ["In: B×256", "Out: B×1", "256→256→256→1"]); out += s_
    c3 = c2 + hc2 + GAP
    s_, hc3 = abx(CX, c3, CW, "State value", ["In: B×1", "used by PPO to compute the advantages"], "io"); out += s_
    out += arrow(CX + CW / 2, fy + hf, CX + CW / 2, ay)
    out += arrow(CX + CW / 2, ay + hc1, CX + CW / 2, c2)
    out += arrow(CX + CW / 2, c2 + hc2, CX + CW / 2, c3)

    H = a5 + ha5 + 84
    out += f'<text x="40" y="{H - 56}" class="s">Total {P_TOT} parameters. B = batch size; 10 jobs, 1 machine, 3 resources, 10 ticks.</text>'
    if single:
        out += f'<text x="40" y="{H - 38}" class="s">Green = the additions over M9: the dynamic machine embedding and the pair features. The fit attention with its feasibility mask is already part of M9.</text>'
    else:
        out += f'<text x="40" y="{H - 38}" class="s">Green = the additions over M9: per-resource fit inputs (and the fit attention that uses them), the dynamic machine embedding and the pair features.</text>'
    out += f'<text x="40" y="{H - 20}" class="s">Every job goes through the same encoder and the same scorer, so the job order does not matter.</text>'
    return svg(W, H, out)


# ---------------------------------------------------------------- E1 network (model M0)
def network_e1_svg():
    """The network used on E1 (every job uses 255): model M0, measured from the first committed version."""
    W = 1100
    out = '<text x="40" y="30" class="h">The network used on E1 (every job uses 255): model M0</text>'

    def bx(x, y, w, title, sub, cls="base"):
        wrap_at = max(20, int(w / 6.6))
        items = sub if isinstance(sub, (list, tuple)) else [sub]
        n = sum(len(wrap(i, wrap_at)) for i in items)
        h = 40 + 16 * n
        return box(x, y, w, h, title, sub, cls, wrap_at=wrap_at), h

    def arrow(x1, y1, x2, y2, cls="arr"):
        return f'<path d="M{x1} {y1} L{x2} {y2}" class="{cls}"/>'

    LX, LW, RX, RW, GAP = 40, 470, 590, 470, 26

    s_, ho = bx(40, 70, 1020, "Observation",
                ["Output: job status, size, ttl, arrival, wait time B×10 each;  time B×1;  (job usage B×10×3×10 and machine arrays B×1×3×10 are not used on E1)"], "io")
    out += s_; ob = 70 + ho
    y = ob + 70; parts = []; pos = []
    def add(title, sub, cls="base"):
        nonlocal y
        s_, h = bx(LX, y, LW, title, sub, cls); parts.append(s_); pos.append((y, h)); y += h + GAP
    add("Per-job features",
        ["In: status B×10, size, wait, ttl, arrival B×10 each", "Out: B×10×20", "status embedding (16 learned numbers) + size, wait, ttl, arrival, scaled by 110 (arrival = ticks until arrival)"])
    add("Job encoder",
        ["In: B×10×20", "Out: B×10×256", "Linear 20→256, LayerNorm, ReLU, twice; the same weights for every job"])
    add("Attention pooling",
        ["In: B×10×256", "Out: global summary B×256",
         "no query, key or value: a small network (256→32→1, tanh) scores how much each job matters, a softmax over the jobs gives weights, the summary is the weighted sum, then Linear 256→256, LayerNorm, ReLU"])
    main_end = y - GAP
    for sp in parts: out += sp
    out += arrow(LX + LW / 2, ob, LX + LW / 2, pos[0][0])
    for i in range(len(pos) - 1):
        out += arrow(LX + LW / 2, pos[i][0] + pos[i][1], LX + LW / 2, pos[i + 1][0])

    s_, hn = bx(RX, pos[0][0], RW, "What this model does not have",
                ["no job self-attention: jobs are never compared with each other, except through the pooled global summary",
                 "no usage or fit inputs: on E1 every job needs the whole machine, so only one job can run at a time and fit is never a question",
                 "no queue_pressure feature (added later in M1)"], "io")
    out += s_

    fy = main_end + 60
    s_, hf = bx(40, fy, 1020, "Feature vector",
                ["In: global summary 256, 10 job embeddings 2560 (straight from the job encoder)", "Out: B×2816   (the actor reads all of it, the critic only the first 256 numbers)"], "base")
    out += s_
    out += f'<rect x="26" y="{ob + 10}" width="1048" height="{fy - 14 - ob - 10}" rx="14" fill="none" stroke="#8c959f" stroke-width="1.4" stroke-dasharray="7 5"/>'
    out += f'<text x="300" y="{ob + 30}" class="s">Shared feature extractor (146,833 parameters), used by the actor and the critic</text>'
    out += arrow(LX + LW / 2, main_end, LX + LW / 2, fy)
    ye, he = pos[1]
    out += f'<path d="M{LX + LW} {ye + he / 2} L{LX + LW + 40} {ye + he / 2} L{LX + LW + 40} {fy}" class="arr"/>'
    out += f'<text x="{LX + LW + 48}" y="{ye + he / 2 - 6}" class="s">job embeddings</text>'

    ay = fy + hf + 44
    out += f'<text x="40" y="{ay - 14}" class="t">Actor head (131,842 parameters)</text>'
    out += f'<text x="880" y="{ay - 14}" class="t" text-anchor="end">Critic (131,841 parameters)</text>'
    AX, AW = 40, 640
    s_, h1 = bx(AX, ay, AW, "Scoring input, one row per job",
                ["In: global B×256, static machine embedding B×1×256 (256 learned numbers), job embeddings B×10×256", "Out: B×10×768   (256 + 256 + 256 for each job)"]); out += s_
    a2 = ay + h1 + GAP
    s_, h2 = bx(AX, a2, 310, "Score network", ["In: B×10×768", "Out: job scores B×10", "768→128→128→1, the same weights for every job"]); out += s_
    s_, h2b = bx(AX + 330, a2, 310, "Skip head", ["In: global summary B×256", "Out: skip score B×1", "256→64→1"]); out += s_
    a3 = a2 + max(h2, h2b) + GAP
    s_, h3 = bx(AX, a3, AW, "Action logits", ["In: skip score B×1, job scores B×10", "Out: B×11   [ skip, job 1 ... job 10 ]"], "io"); out += s_
    a4 = a3 + h3 + GAP
    s_, h4 = bx(AX, a4, AW, "Policy", ["In: logits B×11", "Out: one action per observation, 0 to 10", "softmax, then sample (training) or arg max (evaluation); 0 = skip, k = start job k − 1 on the machine"], "io"); out += s_
    out += arrow(AX + AW / 2, fy + hf, AX + AW / 2, ay)
    out += arrow(AX + 155, ay + h1, AX + 155, a2)
    out += arrow(AX + 485, ay + h1, AX + 485, a2)
    out += arrow(AX + 155, a2 + h2, AX + 155, a3)
    out += arrow(AX + 485, a2 + h2b, AX + 485, a3)
    out += arrow(AX + AW / 2, a3 + h3, AX + AW / 2, a4)

    CX, CW = 730, 330
    s_, c1 = bx(CX, ay, CW, "Global summary", ["In: feature vector B×2816", "Out: B×256 (its first 256 numbers)"]); out += s_
    c2 = ay + c1 + GAP
    s_, c2h = bx(CX, c2, CW, "Value network", ["In: B×256", "Out: B×1", "256→256→256→1"]); out += s_
    c3 = c2 + c2h + GAP
    s_, c3h = bx(CX, c3, CW, "State value", ["In: B×1", "used by PPO to compute the advantages"], "io"); out += s_
    out += arrow(CX + CW / 2, fy + hf, CX + CW / 2, ay)
    out += arrow(CX + CW / 2, ay + c1, CX + CW / 2, c2)
    out += arrow(CX + CW / 2, c2 + c2h, CX + CW / 2, c3)

    H = a4 + h4 + 84
    out += f'<text x="40" y="{H - 56}" class="s">Total 410,516 parameters. B = batch size; 10 jobs, 1 machine. Measured from the first committed version (the np7aenua / 6b195fqa architecture).</text>'
    out += f'<text x="40" y="{H - 38}" class="s">M1 (g231w4jg, the best model on E1) adds a queue_pressure feature and masked job self-attention between the encoder and the pooling (experiment log #6 and #8).</text>'
    out += f'<text x="40" y="{H - 20}" class="s">Every job goes through the same encoder and the same scorer, so the job order does not matter.</text>'
    return svg(W, H, out)


# ---------------------------------------------------------------- bigger network (embedding size 512, 3 resources)
def network_big_svg():
    """Same network as network_full_svg() with the embedding size doubled from 256 to 512.
    Sizes were measured by building the policy with embedding_dim=512. Only the text inside
    <text> elements is rewritten, never the drawing coordinates."""
    import re
    global HEAD_DIM
    HEAD_DIM = 128
    try:
        t = network_full_svg(single=False)
    finally:
        HEAD_DIM = 64
    pairs = (
        ("The current network: from the observation to the action and the value",
         "The bigger network (embedding size 512), 3 resources: from the observation to the action and the value"),
        ("626,513", "2,283,345"), ("140,674", "255,618"), ("131,841", "197,377"), ("899,028", "2,736,340"),
        ("256→256→256→1", "@@VAL@@"),
        ("3162", "6234"), ("832", "1600"), ("320", "576"),
        ("256", "512"),
        ("@@VAL@@", "512→256→256→1"),
        ("Total 2,736,340 parameters. B = batch size;", "Total 2,736,340 parameters (3 times the 256-size network). B = batch size;"),
        ("Every job goes through the same encoder and the same scorer, so the job order does not matter.",
         "Embedding size 512 instead of 256. The fit networks (64), scorer hidden size (128), pooling hidden size (32), 4 heads and 1 attention layer are unchanged."),
    )
    used = set()

    def fix(m):
        body = m.group(2)
        for old, new_ in pairs:
            if old in body:
                used.add(old)
                body = body.replace(old, new_)
        return m.group(1) + body + m.group(3)

    t = re.sub(r"(<text[^>]*>)([^<]*)(</text>)", fix, t)
    missing = [o for o, _ in pairs if o not in used and o != "@@VAL@@"]
    assert not missing, missing
    return t


# ---------------------------------------------------------------- network without attention (DeepSets style)
def network_noattn_svg():
    """The no-attention model (NoAttentionSchedulingPolicy). Sizes measured by building the policy."""
    W = 1100
    out = ('<text x="40" y="30" class="h">The network without attention (DeepSets style): from the observation to the action and the value</text>'
           + legend(40, 44))

    def bx(x, y, w, title, sub, cls="base"):
        wrap_at = max(20, int(w / 6.6))
        items = sub if isinstance(sub, (list, tuple)) else [sub]
        n = sum(len(wrap(i, wrap_at)) for i in items)
        h = 40 + 16 * n
        return box(x, y, w, h, title, sub, cls, wrap_at=wrap_at), h

    def arrow(x1, y1, x2, y2, cls="arr"):
        return f'<path d="M{x1} {y1} L{x2} {y2}" class="{cls}"/>'

    LX, LW, RX, RW, GAP = 40, 470, 590, 470, 26

    s_, ho = bx(40, 70, 1020, "Observation",
                ["Output: machine usage and capacity B×1×3×10;  job usage B×10×3×10;  status, size, ttl, arrival, wait, start and finish times B×10;  time B×1"], "io")
    out += s_; ob = 70 + ho

    # side column first (inputs that feed the per-job vector)
    sy = ob + 70
    s_, h1 = bx(RX, sy, RW, "Usage and room per resource",
                ["In: job usage B×10×3×10, machine usage and capacity B×1×3×10",
                 "Out: job usage B×10×3 (max over time ÷ 255), machine room B×1×3 (min of capacity − usage ÷ 255)"], "new")
    out += s_
    py = sy + h1 + 22
    s_, h2 = bx(RX, py, RW, "Pair features (machine, job)",
                ["In: free grid, job usage and size, status", "Out: B×1×10×9",
                 "fits now, ticks until it fits, finish time, headroom per resource (3), dominant share, packing score, pending flag; computed, no learned weights"], "new")
    out += s_
    dy = py + h2 + 22
    s_, h3 = bx(RX, dy, RW, "Dynamic machine embedding",
                ["In: free grid B×1×3×10 (capacity − usage, ÷ 255)", "Out: B×1×256",
                 "row encoder 10→64→64 for each resource, join the 3 rows (192), Linear 192→256, ReLU, Linear 256→256; the last layer starts at zero"], "new")
    out += s_
    side_end = dy + h3

    # main column
    y = sy; parts = []; pos = []
    def add(title, sub, cls="base"):
        nonlocal y
        s_, h = bx(LX, y, LW, title, sub, cls); parts.append(s_); pos.append((y, h)); y += h + GAP
    add("Per-job input vector",
        ["In: status B×10, size, wait, ttl, arrival B×10 each; job usage B×10×3; machine room B×1×3 (copied to every job); pair features B×10×9",
         "Out: B×10×35", "status embedding (16 learned numbers) + 4 scaled numbers + 3 usage + 3 room + 9 pair = 35"])
    add("Job encoder",
        ["In: B×10×35", "Out: B×10×256", "Linear 35→256, LayerNorm, ReLU, Linear 256→256, LayerNorm, ReLU; the same weights for every job"])
    add("Mean and max pooling over the jobs",
        ["In: B×10×256 (jobs that are not finished)", "Out: B×512 (mean 256 + max 256)",
         "no attention and no learned weights: every unfinished job counts the same in the mean, and the max keeps the strongest signal per channel"])
    add("Global head",
        ["In: B×512", "Out: global summary B×256", "Linear 512→256, LayerNorm, ReLU"])
    main_end = y - GAP
    for sp in parts: out += sp
    out += arrow(LX + LW / 2, ob, LX + LW / 2, pos[0][0])
    for i in range(len(pos) - 1):
        out += arrow(LX + LW / 2, pos[i][0] + pos[i][1], LX + LW / 2, pos[i + 1][0])
    out += arrow(RX + RW / 2, ob, RX + RW / 2, sy)
    # side boxes feed the per-job vector
    out += arrow(RX, sy + h1 / 2, LX + LW, pos[0][0] + pos[0][1] / 2 - 8, "sarr")
    out += arrow(RX, py + h2 / 2, LX + LW, pos[0][0] + pos[0][1] / 2 + 8, "sarr")

    fy = max(main_end, side_end) + 60
    s_, hf = bx(40, fy, 1020, "Feature vector",
                ["In: global summary 256, 10 job embeddings 2560, pair features 90, dynamic machine embedding 256", "Out: B×3162   (the actor reads all of it, the critic only the first 256 numbers)"], "base")
    out += s_
    out += f'<rect x="26" y="{ob + 10}" width="1048" height="{fy - 14 - ob - 10}" rx="14" fill="none" stroke="#8c959f" stroke-width="1.4" stroke-dasharray="7 5"/>'
    out += f'<text x="300" y="{ob + 30}" class="s">Shared feature extractor (328,016 parameters), used by the actor and the critic</text>'
    out += arrow(LX + LW / 2, main_end, LX + LW / 2, fy)
    ye, he = pos[1]
    out += f'<path d="M{LX + LW} {ye + he / 2} L{LX + LW + 36} {ye + he / 2} L{LX + LW + 36} {fy}" class="arr"/>'
    out += f'<text x="{LX + LW + 44}" y="{ye + he / 2 - 6}" class="s">jobs</text>'
    out += f'<path d="M{RX} {dy + h3 / 2} L{LX + LW + 76} {dy + h3 / 2} L{LX + LW + 76} {fy}" class="arr"/>'
    out += arrow(RX + RW - 60, dy + h3, RX + RW - 60, fy)
    out += f'<path d="M{RX + RW} {py + h2 / 2} L{RX + RW + 8} {py + h2 / 2} L{RX + RW + 8} {fy}" class="arr"/>'

    ay = fy + hf + 44
    out += f'<text x="40" y="{ay - 14}" class="t">Actor head (140,674 parameters)</text>'
    out += f'<text x="880" y="{ay - 14}" class="t" text-anchor="end">Critic (131,841 parameters)</text>'
    AX, AW = 40, 640
    s_, a1 = bx(AX, ay, AW, "Machine embedding and pair projection",
                ["In: dynamic machine embedding B×1×256, pair features B×1×10×9", "Out: machine B×1×256, pair B×1×10×64",
                 "static machine embedding + the dynamic one; pair features Linear 9→64, ReLU"]); out += s_
    a2 = ay + a1 + GAP
    s_, a2h = bx(AX, a2, AW, "Scoring input, one row per job",
                 ["In: global B×256, machine B×1×256, job embeddings B×10×256, pair B×10×64", "Out: B×10×832   (256 + 256 + 256 + 64 for each job)"]); out += s_
    a3 = a2 + a2h + GAP
    s_, a3h = bx(AX, a3, 310, "Score network", ["In: B×10×832", "Out: job scores B×10", "832→128→128→1, the same weights for every job"]); out += s_
    s_, a3b = bx(AX + 330, a3, 310, "Skip head", ["In: global summary B×256", "Out: skip score B×1", "256→64→1"]); out += s_
    a4 = a3 + max(a3h, a3b) + GAP
    s_, a4h = bx(AX, a4, AW, "Action logits", ["In: skip score B×1, job scores B×10", "Out: B×11   [ skip, job 1 ... job 10 ]"], "io"); out += s_
    a5 = a4 + a4h + GAP
    s_, a5h = bx(AX, a5, AW, "Policy", ["In: logits B×11", "Out: one action per observation, 0 to 10", "softmax, then sample (training) or arg max (evaluation); 0 = skip, k = start job k − 1 on the machine"], "io"); out += s_
    out += arrow(AX + AW / 2, fy + hf, AX + AW / 2, ay)
    out += arrow(AX + AW / 2, ay + a1, AX + AW / 2, a2)
    out += arrow(AX + 155, a2 + a2h, AX + 155, a3)
    out += arrow(AX + 485, a2 + a2h, AX + 485, a3)
    out += arrow(AX + 155, a3 + a3h, AX + 155, a4)
    out += arrow(AX + 485, a3 + a3b, AX + 485, a4)
    out += arrow(AX + AW / 2, a4 + a4h, AX + AW / 2, a5)

    CX, CW = 730, 330
    s_, c1 = bx(CX, ay, CW, "Global summary", ["In: feature vector B×3162", "Out: B×256 (its first 256 numbers)"]); out += s_
    c2 = ay + c1 + GAP
    s_, c2h = bx(CX, c2, CW, "Value network", ["In: B×256", "Out: B×1", "256→256→256→1"]); out += s_
    c3 = c2 + c2h + GAP
    s_, c3h = bx(CX, c3, CW, "State value", ["In: B×1", "used by PPO to compute the advantages"], "io"); out += s_
    out += arrow(CX + CW / 2, fy + hf, CX + CW / 2, ay)
    out += arrow(CX + CW / 2, ay + c1, CX + CW / 2, c2)
    out += arrow(CX + CW / 2, c2 + c2h, CX + CW / 2, c3)

    H = a5 + a5h + 84
    out += f'<text x="40" y="{H - 56}" class="s">Total 600,531 parameters (the attention model has 899,028). B = batch size; 10 jobs, 1 machine, 3 resources, 10 ticks.</text>'
    out += f'<text x="40" y="{H - 38}" class="s">No attention anywhere: no fit attention, no job self-attention, no attention pooling. The fit information comes in as plain inputs (usage, room, pair features).</text>'
    out += f'<text x="40" y="{H - 20}" class="s">Green = inputs that the attention model handled with attention. Every job goes through the same encoder and scorer, so the job order does not matter.</text>'
    return svg(W, H, out)


def main():
    for k in MODELS:
        (OUT_DIR / f"model_{k}.svg").write_text(model_svg(k))
    (OUT_DIR / "environments.svg").write_text(env_svg())
    (OUT_DIR / "results_E4.svg").write_text(results_svg())
    (OUT_DIR / "lineage.svg").write_text(lineage_svg())
    (OUT_DIR / "model_pfrw14vh.svg").write_text(pfrw_svg())
    (OUT_DIR / "network_full.svg").write_text(network_full_svg())
    (OUT_DIR / "network_full_single_usage.svg").write_text(network_full_svg(single=True))
    (OUT_DIR / "network_e1_m0.svg").write_text(network_e1_svg())
    (OUT_DIR / "network_full_big.svg").write_text(network_big_svg())
    (OUT_DIR / "network_no_attention.svg").write_text(network_noattn_svg())
    print("wrote", sorted(p.name for p in OUT_DIR.glob("*.svg")))


if __name__ == "__main__":
    main()
