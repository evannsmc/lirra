#!/bin/bash
# DAgger + margin-weighted BC + PPO pipeline for one (system, spec), the recipe that worked on the blimp:
#
#   1. DAgger rounds driven by the original-loss BC (each round retrains BC with the original loss,
#      warm-started); 80% policy-driven episodes with noise on the first input, 20% expert;
#   2. margin-weighted BC (eps-cert 0.05, lambda-cert 5, 300 epochs) on the original data alone and
#      on the original + DAgger data;
#   3. PPO from the margin + DAgger BC with reward B (price rejections + margin 0.05) and
#      C (B + checked-loop fallback with a synthesis price);
#   4. shield and policy_only evaluation of the two margin BCs and the two PPO policies.
#
#   scripts/dagger_ppo_pipeline.sh <system> <spec> <bc0 policy.npz> <base dataset.npz>
#
# Environment: ROUNDS (2), EPISODES (300), WORKERS (7), ROUND_EPOCHS (100), TAG (dg),
# PPO_STEPS (50000000), EPISODE_STEPS (120; 240 for the double integrator), LOGDIR (runs/pipeline_logs),
# TARGET_TOTAL (unset: EPISODES per round; set: episodes of each round chosen so that the base data plus
# all rounds reach about TARGET_TOTAL labels, split evenly over the remaining rounds, using the labels per
# episode of the previous round, RATE0 (45) for the first), SKIP_EVALS (1: no evaluations here).
# Runs from the repository root on a GPU node (BC and PPO on the GPU, datasets and evaluations on CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
sys=$1; spec=$2; bc=$3; data=("$4")
ROUNDS=${ROUNDS:-2}; EPISODES=${EPISODES:-300}; WORKERS=${WORKERS:-7}; ROUND_EPOCHS=${ROUND_EPOCHS:-100}
TAG=${TAG:-dg}; PPO_STEPS=${PPO_STEPS:-50000000}
EPISODE_STEPS=${EPISODE_STEPS:-$([[ $sys == di ]] && echo 240 || echo 120)}
LOGDIR=${LOGDIR:-runs/pipeline_logs}; mkdir -p "$LOGDIR"
unset PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cpu() { RL_STL_DEVICE=cpu JAX_PLATFORMS=cpu "$@"; }
gpu() { RL_STL_DEVICE=gpu "$@"; }
latest() { ls -d $1 2>/dev/null | sort | tail -1; }
labels() { uv run --no-sync python -c "import numpy as np, sys; print(sum(len(np.load(f, allow_pickle=True)['status']) for f in sys.argv[1:]))" "$@"; }
RATE=${RATE0:-45}
MARGIN="--eps-cert 0.05 --lambda-cert 5 --epochs 300"
REWARD_B="--c-cert 100 --c-ok 10 --eps-cert 0.05"
REWARD_C="$REWARD_B --fallback --c-syn 10"

evals() {  # <policy> <tag>: shield and policy_only in the background
    [[ ${SKIP_EVALS:-0} == 1 ]] && return 0
    for pair in shield policy_only; do
        (set +e; cpu uv run --no-sync python experiments/patrol/evaluate.py --system $sys --spec $spec --seeds 10 \
            $([[ $sys == di ]] && echo "--levels 0 2 4") --pairs $pair --policy "$1" --reference clock \
            --tag ${spec}_$2_$pair > "$LOGDIR/ev_${sys}_${spec}_$2_$pair.log" 2>&1
         echo "eval done $2 $pair $?") &
    done
}

for r in $(seq 1 "$ROUNDS"); do
    if [[ -n ${TARGET_TOTAL:-} ]]; then
        have=$(labels "${data[@]}")
        EPISODES=$(python3 -c "import math; print(max(22, math.ceil(($TARGET_TOTAL - $have) / ($ROUNDS - $r + 1) / $RATE)))")
        echo "round $r: $have labels so far, target $TARGET_TOTAL, rate $RATE labels/episode -> $EPISODES episodes"
    fi
    echo "round $r: dataset driven by $bc"
    args=(experiments/patrol/dataset.py --system $sys --spec $spec --reference clock --episodes $EPISODES
          --seed $((20 + r)) --policy "$bc" --mix 0.2 0 0 0.8 --policy-noise 0 0.05 0.1 --tag ${spec}_${TAG}_r$r)
    cpu uv run --no-sync python "${args[@]}" --workers $WORKERS
    ds=$(latest "runs/patrol_dataset/$sys/*_${spec}_${TAG}_r$r")
    [[ -f $ds/summary.json ]] || { echo "dataset round $r failed"; exit 1; }
    echo "dataset $ds $(tr -d '\n ' < "$ds/summary.json")"
    data+=("$ds/dataset.npz")
    RATE=$(python3 -c "print($(labels "$ds/dataset.npz") / $EPISODES)")
    gpu uv run --no-sync python experiments/patrol/train_bc.py --system $sys --spec $spec --reference clock \
        --data "${data[@]}" --init "$bc" --epochs $ROUND_EPOCHS --tag ${spec}_${TAG}_r$r
    bc=$(latest "runs/patrol_bc/$sys/*_${spec}_${TAG}_r$r")/policy.npz
    echo "bc $bc"
done

echo "margin-weighted BC"
gpu uv run --no-sync python experiments/patrol/train_bc.py --system $sys --spec $spec --reference clock \
    --data "${data[0]}" $MARGIN --tag ${spec}_${TAG}_margin_base &
gpu uv run --no-sync python experiments/patrol/train_bc.py --system $sys --spec $spec --reference clock \
    --data "${data[@]}" $MARGIN --tag ${spec}_${TAG}_margin_all &
wait
mbase=$(latest "runs/patrol_bc/$sys/*_${spec}_${TAG}_margin_base")/policy.npz
mall=$(latest "runs/patrol_bc/$sys/*_${spec}_${TAG}_margin_all")/policy.npz
echo "bc margin $mbase $mall"
evals "$mbase" ${TAG}_margin_base_bc
evals "$mall" ${TAG}_margin_all_bc

for v in B C; do
    flags=REWARD_$v
    gpu uv run --no-sync python experiments/patrol/train_ppo.py --system $sys --spec $spec --init "$mall" \
        --num-timesteps $PPO_STEPS --num-envs 2048 --batch-size 1024 --num-evals 26 --device gpu \
        --episode-steps $EPISODE_STEPS ${!flags} --tag ${spec}_${TAG}_ppo$v &
done
wait
for v in B C; do
    ppo=$(latest "runs/patrol_ppo/$sys/*_${spec}_${TAG}_ppo$v")/policy.npz
    echo "ppo $v $ppo"
    evals "$ppo" ${TAG}_ppo$v
done
wait
echo "pipeline done $sys $spec"
