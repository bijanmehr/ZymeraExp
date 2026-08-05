"""Build a self-contained CONNECTIVITY-FIRST HTML audit report for the greedy->MVProp swap.

Reads each run's history.json (per-iter metrics) + optional GIFs, and emits one HTML file with
inline SVG training curves (connectivity_real leads; coverage second) + a greedy-vs-mvprop table
+ base64-embedded training/inference GIFs. No external assets — opens anywhere.

    $PY -m t5lab.make_mvprop_report --out report/mvprop_report.html \
        --run "open 24² greedy=runs/mvprop/open24_greedy" \
        --run "open 24² mvprop=runs/mvprop/open24_mvprop" \
        --gif "open 24² mvprop=gifs/mvprop_open24_s0.gif,gifs/mvprop_open24_s3.gif"
"""
import argparse
import base64
import json
import os

_CSS = """
:root{--bg:#f6f4ee;--ink:#20242c;--soft:#5c6270;--faint:#8b8f9a;--rule:#e2ddd0;--card:#fbfaf5;
--bad:#b5462f;--ok:#2f6d57;--warn:#a5762a;--conn:#2f6d57;--cov:#3b6ea5}
@media(prefers-color-scheme:dark){:root{--bg:#14161b;--ink:#e8eaef;--soft:#aeb4c0;--faint:#767c8a;
--rule:#2b2f38;--card:#1b1e25;--bad:#e0714f;--ok:#5cc9a0;--warn:#d7a54a;--conn:#5cc9a0;--cov:#6fa8dc}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,Arial;margin:0}
.wrap{max-width:1000px;margin:0 auto;padding:34px 22px 70px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--soft);margin:0 0 16px}h2{font-size:17px;margin:28px 0 8px}
.banner{border-left:4px solid var(--ok);background:var(--card);padding:14px 18px;border-radius:0 8px 8px 0;margin:14px 0}
.banner.bad{border-left-color:var(--bad)}
table{border-collapse:collapse;font-size:13.5px;width:100%;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--rule);padding:6px 10px;text-align:center}th{background:var(--card)}td.l,th.l{text-align:left}
.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok);font-weight:600}.warn{color:var(--warn);font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:14px 0}
.cell{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px}
.cell img{width:100%;border-radius:6px;background:#000;image-rendering:pixelated}
.cap{font-size:13px;color:var(--soft);margin-top:8px}small{color:var(--faint)}
.chart{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:10px 12px;margin:10px 0}
.legend{font-size:12px;color:var(--soft)}.legend b.c{color:var(--conn)}.legend b.v{color:var(--cov)}
"""


def _series(hist, key):
    return [h.get(key) for h in hist if h.get(key) is not None]


def _tail_mean(hist, key, n=10):
    xs = [h.get(key) for h in hist[-n:] if h.get(key) is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _peak(hist, key):
    xs = [h.get(key) for h in hist if h.get(key) is not None]
    return max(xs) if xs else float("nan")


def _svg_lines(hist, w=460, h=150, pad=28):
    """Two overlaid lines over iterations: connectivity_real (green) + coverage_pct (blue),
    both on a 0..1 axis, with a dashed 0.90 connectivity target line."""
    cr = _series(hist, "connectivity_real")
    cv = _series(hist, "coverage_pct")
    n = max(len(cr), len(cv), 2)

    def path(ys, color):
        if len(ys) < 2:
            return ""
        step = max(1, len(ys) // 400)
        ys = ys[::step]
        m = len(ys)
        pts = []
        for i, y in enumerate(ys):
            x = pad + (w - 2 * pad) * i / (m - 1)
            yy = (h - pad) - (h - 2 * pad) * max(0.0, min(1.0, y))
            pts.append(f"{x:.1f},{yy:.1f}")
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>'

    y90 = (h - pad) - (h - 2 * pad) * 0.90
    axis = (f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="var(--rule)"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="var(--rule)"/>'
            f'<line x1="{pad}" y1="{y90:.1f}" x2="{w-pad}" y2="{y90:.1f}" stroke="var(--ok)" '
            f'stroke-dasharray="4 4" stroke-width="1" opacity="0.6"/>'
            f'<text x="{w-pad-2}" y="{y90-3:.1f}" text-anchor="end" font-size="10" '
            f'fill="var(--ok)">conn target 0.90</text>'
            f'<text x="{pad-4}" y="{pad+4}" text-anchor="end" font-size="10" fill="var(--faint)">1.0</text>'
            f'<text x="{pad-4}" y="{h-pad}" text-anchor="end" font-size="10" fill="var(--faint)">0</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">'
            f'{axis}{path(cr, "var(--conn)")}{path(cv, "var(--cov)")}</svg>')


def _b64_img(path):
    if not path or not os.path.exists(path):
        return f'<div class="cap">[missing: {path}]</div>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/gif;base64,{b}" alt="rollout">'


def build(runs, gifs, out, title="MVProp planner — connectivity-first audit", interim=False):
    """runs: list of (label, run_dir). gifs: dict label -> [gif paths]. interim=True frames a
    mid-training snapshot (recent values, no hard pass/fail)."""
    data = []
    for label, rd in runs:
        hp = os.path.join(rd, "history.json")
        hist = json.load(open(hp)) if os.path.exists(hp) else []
        data.append((label, rd, hist))

    mv = [(l, h) for (l, rd, h) in data if "mvprop" in l.lower() and h]
    if interim:
        best = max(mv, key=lambda x: _tail_mean(x[1], "coverage_pct"), default=(None, None))
        bl, bh = best
        note = (f'best mvprop arm so far — <b>{bl}</b>: connectivity_real '
                f'<b>{_tail_mean(bh, "connectivity_real"):.3f}</b>, coverage '
                f'<b>{_tail_mean(bh, "coverage_pct")*100:.1f}%</b> (peak {_peak(bh, "coverage_pct")*100:.1f}%).'
                if bh else 'no data yet.')
        banner = (f'<div class="banner"><b>MID-TRAINING SNAPSHOT — not converged.</b> Values are the '
                  f'mean of the last ~10 logged points (recent), not a final result; arms still climbing. '
                  f'{note} GIFs land when a run completes and checkpoints.</div>')
    else:
        worst_conn = min([_tail_mean(h, "connectivity_real") for _, h in mv], default=float("nan"))
        ok = worst_conn >= 0.90
        banner = (f'<div class="banner {"" if ok else "bad"}"><b>{"PASS" if ok else "CONNECTIVITY FAILURE"} '
                  f'— connectivity-first.</b> Worst mvprop connectivity_real (λ₂&gt;0.5) across worlds = '
                  f'<b>{worst_conn:.3f}</b> (target ≥0.90). Coverage graded only on runs whose graph holds.</div>')

    # comparison table (recent = mean of last ~10 points; peak = best seen)
    rows = ""
    for label, rd, hist in data:
        if not hist:
            rows += f'<tr><td class="l">{label}</td><td colspan="6"><small>no history yet ({rd})</small></td></tr>'
            continue
        cr = _tail_mean(hist, "connectivity_real"); cp = _tail_mean(hist, "connectivity_pct")
        cv = _tail_mean(hist, "coverage_pct"); pv = _peak(hist, "coverage_pct"); dl = _tail_mean(hist, "dual_lambda")
        crc = "ok" if cr >= 0.90 else ("warn" if cr >= 0.7 else "bad")
        rows += (f'<tr><td class="l">{label}</td>'
                 f'<td class="{crc}">{cr:.3f}</td><td>{cp:.3f}</td>'
                 f'<td>{cv*100:.1f}%</td><td>{pv*100:.1f}%</td><td>{dl:.2f}</td><td>{len(hist)}</td></tr>')
    table = (f'<table><tr><th class="l">arm</th><th>CONN_real ↑<br><small>λ₂&gt;0.5 (recent)</small></th>'
             f'<th>conn<br><small>λ₂&gt;1e-3</small></th><th>cov<br><small>recent</small></th>'
             f'<th>cov<br><small>peak</small></th><th>dual λ</th><th>log pts</th></tr>'
             f'{rows}</table>')

    # charts + gifs per arm
    charts = ""
    for label, rd, hist in data:
        if not hist:
            continue
        charts += (f'<h2>{label}</h2><div class="chart">{_svg_lines(hist)}'
                   f'<div class="legend"><b class="c">■</b> connectivity_real &nbsp; '
                   f'<b class="v">■</b> coverage &nbsp;(x = iterations)</div></div>')
        gl = gifs.get(label, [])
        if gl:
            cells = "".join(f'<div class="cell">{_b64_img(g)}<div class="cap">{os.path.basename(g)}</div></div>'
                            for g in gl)
            charts += f'<div class="grid">{cells}</div>'

    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>{_CSS}</style></head><body><div class="wrap">'
            f'<h1>{title}</h1>'
            f'<p class="sub">Exact previous architecture (setpool central critic · lagrangian dual · '
            f'local_edge_margin · reach reward · λ̂₂ aux · collision-mask · no connectivity hard-mask), '
            f'greedy → distilled MVProp planner. One-variable swap.</p>'
            f'{banner}<h2>Summary (recent = mean of last ~10 logged points)</h2>{table}{charts}</div></body></html>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(html)//1024} KB)", flush=True)
    return out


def _kv(s):
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--run", action="append", default=[], help='"label=run_dir"')
    p.add_argument("--gif", action="append", default=[], help='"label=g1.gif,g2.gif"')
    a = p.parse_args(argv)
    runs = [_kv(r) for r in a.run]
    gifs = {k: v.split(",") for k, v in (_kv(g) for g in a.gif)}
    build(runs, gifs, a.out)


if __name__ == "__main__":
    main()
