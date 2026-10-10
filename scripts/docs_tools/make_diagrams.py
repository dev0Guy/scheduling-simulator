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


def main():
    for k in MODELS:
        (OUT_DIR / f"model_{k}.svg").write_text(model_svg(k))
    (OUT_DIR / "environments.svg").write_text(env_svg())
    (OUT_DIR / "results_E4.svg").write_text(results_svg())
    (OUT_DIR / "lineage.svg").write_text(lineage_svg())
    (OUT_DIR / "model_pfrw14vh.svg").write_text(pfrw_svg())
    print("wrote", sorted(p.name for p in OUT_DIR.glob("*.svg")))


if __name__ == "__main__":
    main()
