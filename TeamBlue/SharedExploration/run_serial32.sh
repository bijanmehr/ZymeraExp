#!/usr/bin/env bash
# run_serial32.sh — safe single-card launch: 32² units one-at-a-time (zheavy), light 16²/24²
# units packed in parallel (zlight). Peak ≈ 1×50GB + 3×9GB ≈ 77GB < 96GB, and there is never
# more than one 32² training on the card. Run ON BALTHAR from .../TeamBlue/SharedExploration:
#     bash run_serial32.sh
set -uo pipefail

PY="${PY:-$HOME/ZymeraLab/.venv/bin/python}"
PP=".:../../../FiedlerValueEstimation"
HERE="$(pwd)"
SEEDS="${SEEDS:-3}"
ROLLOUTS="${ROLLOUTS:-16}"        # Option B: keep rollouts=16 (exact spec, no experimental change)
JLIGHT="${JLIGHT:-3}"            # light-tier concurrency (3 → ~27GB alongside the 50GB heavy)

mkdir -p runs/roles_t2 runs/lag_t3 runs/critic_t4 report

mk() {   # $1 session, $2 command
  tmux kill-session -t "$1" 2>/dev/null || true
  tmux new-session -d -s "$1" \
    "cd '$HERE'; export PY='$PY' PYTHONPATH='$PP' XLA_PYTHON_CLIENT_PREALLOCATE=false; $2; \
     echo; echo \"=== $1 FINISHED ===\"; exec bash"
  echo "  [$1] launched   (attach: tmux attach -t $1)"
}

echo "=== serial-32 launch: zheavy (32², one-at-a-time) + zlight (16²/24², parallel) ==="

# LIGHT tier — every grid<32 unit, packed JLIGHT at a time.
mk zlight "'$PY' -m ctde_v0.run_serial32 --tier light --jobs $JLIGHT --seeds $SEEDS --rollouts $ROLLOUTS 2>&1 | tee runs/serial32_light.log"

# HEAVY tier — every 32² unit, strictly one at a time. When it finishes: wait for the light tier
# to finish too, run the T4 zero-shot transfer eval (needs the 16² source ckpts from the light
# tier), then render all three reports.
mk zheavy "'$PY' -m ctde_v0.run_serial32 --tier heavy --jobs 1 --seeds $SEEDS --rollouts $ROLLOUTS 2>&1 | tee runs/serial32_heavy.log; \
  echo '[zheavy] waiting for light tier to finish before eval+reports...'; \
  for i in \$(seq 1 2880); do grep -q 'SERIAL32 light DONE' runs/serial32_light.log 2>/dev/null && break; sleep 5; done; \
  echo '[zheavy] running T4 zero-shot transfer eval...'; \
  '$PY' -m ctde_v0.run_critic_t4 --only eval --out runs/critic_t4 --seeds $SEEDS 2>&1 | tee -a runs/serial32_heavy.log; \
  echo '[zheavy] rendering reports...'; \
  bash _report_for.sh roles_t2; bash _report_for.sh lag_t3; bash _report_for.sh critic_t4"

cat <<EOF

  monitor:   tmux ls   ·   attach: tmux attach -t zheavy   (zlight;  detach = Ctrl-b d)
  progress:  grep -h '\[launch\]\|\[finish\]' runs/serial32_heavy.log   (32² one-at-a-time)
             grep -h '\[launch\]\|\[finish\]' runs/serial32_light.log   (16²/24² parallel)
  results:   runs/{roles_t2,lag_t3,critic_t4}/…/{model.eqx,config.json,history.json}   (resumable)
  reports:   report/{roles_t2,lag_t3,critic_t4}/index.html   (auto-built when BOTH tiers finish)

  safety:    one 32² (~50GB) + ${JLIGHT}×light (~9GB) ≈ 77GB < 96GB; heavy tier is jobs=1 so
             the card never holds two 32² trainings. If you want the light tier denser: JLIGHT=4.
EOF
