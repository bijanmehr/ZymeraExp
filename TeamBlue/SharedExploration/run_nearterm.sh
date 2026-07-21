#!/usr/bin/env bash
# run_nearterm.sh — launch the near-term experiment screens, one tmux session each, with local
# checkpoints and an auto-built HTML report per experiment.
#
# Run ON BALTHAR from .../TeamBlue/SharedExploration:
#     bash run_nearterm.sh
#
# Each session: runs its runner (checkpoints -> its own runs/<exp>/ dir, resumable) and, when the
# runner finishes, builds report/<exp>/index.html via _report_for.sh.
set -uo pipefail

# ---- config (override via env if needed) ----
PY="${PY:-$HOME/ZymeraLab/.venv/bin/python}"          # balthar venv python
PP=".:../../../FiedlerValueEstimation"                # PYTHONPATH
HERE="$(pwd)"
ITERS_SMALL="${ITERS_SMALL:-1500}"                    # T2 iters
ITERS_BIG="${ITERS_BIG:-2000}"                        # T3/T4 iters @32²
ROLLOUTS="${ROLLOUTS:-16}"

mkdir -p runs/roles_t2 runs/lag_t3 runs/critic_t4 report probes

# JAX reserves ~75% of the GPU per process by default; PREALLOCATE=false lets the 3 GPU sessions
# coexist on ONE card (on-demand alloc). If you have >1 GPU, set GPUS below and each session pins
# to its own card with CUDA_VISIBLE_DEVICES (then the memory fight disappears).
GPUS="${GPUS:-}"                                      # e.g. GPUS="0 1 2" to shard; empty = share one card
_gpu_env() {   # $1 = index into GPUS list; echoes the env prefix
  if [ -n "$GPUS" ]; then
    local arr=($GPUS); local dev="${arr[$(( $1 % ${#arr[@]} ))]}"
    echo "export CUDA_VISIBLE_DEVICES=$dev;"
  else
    echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false;"
  fi
}

mk() {   # $1 session, $2 gpu-index, $3 command (PY expanded literally)
  tmux kill-session -t "$1" 2>/dev/null || true
  tmux new-session -d -s "$1" \
    "cd '$HERE'; export PY='$PY' PYTHONPATH='$PP'; $(_gpu_env "$2") $3; echo; echo \"=== $1 FINISHED ===\"; exec bash"
  echo "  [$1] launched   (attach: tmux attach -t $1)"
}

echo "=== near-term experiment screens -> tmux sessions zt2 / zt3 / zt4 ==="
mk zt2 0 "'$PY' -m ctde_v0.run_roles_t2  --out runs/roles_t2  --iters $ITERS_SMALL --rollouts $ROLLOUTS --jobs 3 2>&1 | tee runs/roles_t2/log.txt; bash _report_for.sh roles_t2"
mk zt3 1 "'$PY' -m ctde_v0.run_lag_t3    --out runs/lag_t3    --seeds 3 --iters $ITERS_BIG --rollouts $ROLLOUTS --jobs 2 2>&1 | tee runs/lag_t3/log.txt; bash _report_for.sh lag_t3"
mk zt4 2 "'$PY' -m ctde_v0.run_critic_t4 --out runs/critic_t4 --jobs 2 2>&1 | tee runs/critic_t4/log.txt; bash _report_for.sh critic_t4"

cat <<EOF

  monitor:   tmux ls   ·   attach: tmux attach -t zt2   (zt3 / zt4;  detach = Ctrl-b d)
  results:   runs/{roles_t2,lag_t3,critic_t4}/…/{model.eqx,config.json,history.json}   (resumable — reruns skip finished dirs)
  reports:   report/{roles_t2,lag_t3,critic_t4}/index.html   (auto-built when each session's runner finishes)

  GPU:  sharing ONE card by default (PREALLOCATE=false). For MULTIPLE cards:  GPUS="0 1 2" bash run_nearterm.sh
        if a 32² session OOMs, in that session drop --jobs to 1  or  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

  T1 probes (CPU — run once zt2 has produced a checkpoint, e.g. runs/roles_t2/seed0/o16/role/model.eqx):
    tmux new -s zt1 "cd '$HERE'; export PYTHONPATH='$PP'; \\
      '$PY' -m ctde_v0.probe_frontier_align   --run-dir runs/roles_t2/seed0/o16/role --out probes/frontier --seeds 0,1,2; \\
      '$PY' -m ctde_v0.probe_message_ablation --run-dir runs/roles_t2/seed0/o16/role --out probes/msg --mode graphoff"
EOF
