#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/lbh/HiVT"
LOG_ROOT="${LOG_ROOT:-/home/lbh/HiVT/logs}"
RUNS_ROOT="${RUNS_ROOT:-/home/lbh/HiVT/runs}"
GPU_ID="${GPU_ID:-2}"
EMBED_DIM="${EMBED_DIM:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_VERSION="${RUN_VERSION:-reliability_$(date +%Y%m%d_%H%M%S)_dim${EMBED_DIM}_gpu${GPU_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hivt_reliability}"

mkdir -p "$LOG_ROOT" "$RUNS_ROOT"

stdout_log="$LOG_ROOT/${RUN_VERSION}.stdout.log"
pid_file="$LOG_ROOT/${RUN_VERSION}.pid"
meta_file="$LOG_ROOT/${RUN_VERSION}.meta"
cmd_pattern="python train.py --root /home/lbh/HiVT/datasets/argoverse --embed_dim ${EMBED_DIM} --gpus 1 --train_batch_size ${TRAIN_BATCH_SIZE} --val_batch_size ${VAL_BATCH_SIZE} --num_workers ${NUM_WORKERS} --experiment_root ${RUNS_ROOT} --experiment_name ${EXPERIMENT_NAME} --experiment_version ${RUN_VERSION} --use_reliability true"

cd "$PROJECT_ROOT"
nohup env \
  RUN_VERSION="$RUN_VERSION" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUNS_ROOT="$RUNS_ROOT" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  bash -lc "exec ./run_single_gpu.sh train_reliability ${EMBED_DIM} ${GPU_ID}" \
  > "$stdout_log" 2>&1 &
pid=$!

python_pid=""
for _ in $(seq 1 30); do
  python_pid="$(pgrep -af "$cmd_pattern" | awk 'NR==1 {print $1}' || true)"
  if [[ -n "$python_pid" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "$python_pid" ]]; then
  python_pid="$pid"
fi

cat > "$meta_file" <<EOF
launcher_pid=$pid
pid=$python_pid
run_version=$RUN_VERSION
experiment_name=$EXPERIMENT_NAME
stdout_log=$stdout_log
pid_file=$pid_file
run_dir=$RUNS_ROOT/$EXPERIMENT_NAME/$RUN_VERSION
cmd_pattern=$cmd_pattern
EOF

echo "$python_pid" > "$pid_file"

echo "launcher_pid=$pid"
echo "pid=$python_pid"
echo "run_version=$RUN_VERSION"
echo "stdout_log=$stdout_log"
echo "pid_file=$pid_file"
echo "run_dir=$RUNS_ROOT/$EXPERIMENT_NAME/$RUN_VERSION"
