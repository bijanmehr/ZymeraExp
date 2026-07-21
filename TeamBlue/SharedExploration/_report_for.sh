#!/usr/bin/env bash
# _report_for.sh <runs-subdir>
#   Build report/<subdir>/index.html (the GIF gallery, like the campaign report) from every
#   checkpoint under runs/<subdir>/. Run from .../TeamBlue/SharedExploration.
set -uo pipefail
SUB="${1:?usage: bash _report_for.sh <runs-subdir>   e.g. roles_t2}"
PY="${PY:-$HOME/ZymeraLab/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.:../../../FiedlerValueEstimation}"

# collect every completed run (has a model.eqx) -> a "tag:dir" pair for make_report
mapfile -t ckpts < <(find "runs/$SUB" -name model.eqx 2>/dev/null | sort)
if [ "${#ckpts[@]}" -eq 0 ]; then
  echo "[report] no checkpoints under runs/$SUB yet — nothing to render."
  exit 0
fi
args=()
for m in "${ckpts[@]}"; do
  d="$(dirname "$m")"
  tag="$(printf '%s' "$d" | sed "s#runs/$SUB/##; s#/#_#g")"   # seed0/o32/role -> seed0_o32_role
  args+=("$tag:$d")
done
echo "[report] rendering ${#args[@]} runs from runs/$SUB -> report/$SUB/ ..."
"$PY" -m ctde_v0.make_report --runs "${args[@]}" --out "report/$SUB"
echo "[report] done -> report/$SUB/index.html"
