#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/lbh/HiVT"
DATA_ROOT="${DATA_ROOT:-/home/lbh/HiVT/datasets/argoverse}"
CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-HiVT}"
RUNS_ROOT="${RUNS_ROOT:-/home/lbh/HiVT/runs}"
LOG_ROOT="${LOG_ROOT:-/home/lbh/HiVT/logs}"

usage() {
  cat <<'EOF'
Usage:
  ./run_single_gpu.sh train [64|128] [gpu_id]
  ./run_single_gpu.sh train_reliability [64|128] [gpu_id]
  ./run_single_gpu.sh train_reliability_shift [64|128] [gpu_id]
  ./run_single_gpu.sh eval <ckpt_path> [gpu_id] [batch_size]

Examples:
  ./run_single_gpu.sh train 64
  ./run_single_gpu.sh train 128 2
  ./run_single_gpu.sh train_reliability 64
  ./run_single_gpu.sh train_reliability_shift 64 2
  ./run_single_gpu.sh eval /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt
  ./run_single_gpu.sh eval /home/lbh/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt 2 32

Environment overrides:
  DATA_ROOT=/path/to/argoverse
  CONDA_ENV=HiVT
  TRAIN_BATCH_SIZE=32
  VAL_BATCH_SIZE=32
  EVAL_BATCH_SIZE=32
  NUM_WORKERS=8
  LIMIT_TRAIN_BATCHES=1.0
  LIMIT_VAL_BATCHES=1.0
  MAX_EPOCHS=64
  SEED=2022
  CKPT_PATH=/path/to/checkpoint.ckpt
  INIT_CKPT_PATH=/path/to/checkpoint.ckpt
  # shift augmentation overrides（仅对 train_reliability_shift 有效）
  SHIFT_HISTORY_DROPOUT_P=0.3
  SHIFT_NEIGHBOR_DROPOUT_P=0.2
  SHIFT_POSITION_NOISE_STD=0.1
  SHIFT_HEADING_NOISE_STD=0.05
  SHIFT_MAP_JITTER_STD=0.05
  SHIFT_LANE_DROPOUT_P=0.1
  MODE_TARGET_POLICY=fde_only
  SCENE_TARGET_POLICY=target_best_mode_fail
  RISK_FDE_THRESHOLD=2.0
  RISK_MISS_THRESHOLD=4.0
  RISK_CONFLICT_THRESHOLD=1.0
  RISK_CONFLICT_MIN_FRAMES=2
  RISK_CONFLICT_SCOPE=target_to_neighbors
  RISK_SCENE_RATE_THRESHOLD=0.5
  RELIABILITY_RERANK_ALPHA=0.0
  SCENE_LOSS_WEIGHT=0.2
  FREEZE_BACKBONE=false
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

ensure_dir() {
  local path="$1"
  mkdir -p "$path"
}

pick_gpu() {
  if [[ $# -ge 1 && -n "${1:-}" ]]; then
    echo "$1"
    return
  fi

  require_file "/usr/bin/nvidia-smi"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2,2nr \
    | head -n1 \
    | cut -d',' -f1 \
    | tr -d ' '
}

check_data_ready() {
  require_dir "$DATA_ROOT/train/data"
  require_dir "$DATA_ROOT/val/data"
  require_dir "$DATA_ROOT/map_files"
}

check_preprocessed_hint() {
  local train_processed="$DATA_ROOT/train/processed"
  local val_processed="$DATA_ROOT/val/processed"

  if [[ ! -d "$train_processed" || ! -d "$val_processed" ]]; then
    echo "Warning: processed directories are missing; training will trigger preprocessing." >&2
    return
  fi

  local train_count val_count
  train_count="$(find "$train_processed" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  val_count="$(find "$val_processed" -maxdepth 1 -type f -name '*.pt' | wc -l)"

  if [[ "$train_count" -eq 0 || "$val_count" -eq 0 ]]; then
    echo "Warning: processed data looks incomplete; training may spend time preprocessing first." >&2
  fi
}

activate_env() {
  require_file "$CONDA_SH"
  local nounset_was_enabled=0
  if [[ $- == *u* ]]; then
    nounset_was_enabled=1
    set +u
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  if [[ "$nounset_was_enabled" -eq 1 ]]; then
    set -u
  fi
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
}

run_train() {
  local embed_choice="${1:-64}"
  local gpu_id
  gpu_id="$(pick_gpu "${2:-}")"

  local embed_dim
  case "$embed_choice" in
    64|128)
      embed_dim="$embed_choice"
      ;;
    *)
      echo "Unsupported model size: $embed_choice (expected 64 or 128)" >&2
      exit 1
      ;;
  esac

  local train_batch_size="${TRAIN_BATCH_SIZE:-32}"
  local val_batch_size="${VAL_BATCH_SIZE:-32}"
  local num_workers="${NUM_WORKERS:-8}"
  local run_version="${RUN_VERSION:-train_$(date +%Y%m%d_%H%M%S)_dim${embed_dim}_gpu${gpu_id}}"
  local experiment_name="${EXPERIMENT_NAME:-hivt_base}"

  check_data_ready
  check_preprocessed_hint
  activate_env
  ensure_dir "$RUNS_ROOT"
  ensure_dir "$LOG_ROOT"

  cd "$PROJECT_ROOT"
  echo "Using GPU $gpu_id for training"
  echo "embed_dim=$embed_dim train_batch_size=$train_batch_size val_batch_size=$val_batch_size num_workers=$num_workers"
  echo "experiment_name=$experiment_name run_version=$run_version"

  exec env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python train.py \
    --root "$DATA_ROOT" \
    --embed_dim "$embed_dim" \
    --gpus 1 \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --num_workers "$num_workers" \
    --experiment_root "$RUNS_ROOT" \
    --experiment_name "$experiment_name" \
    --experiment_version "$run_version"
}

run_train_reliability() {  local embed_choice="${1:-64}"
  local gpu_id
  gpu_id="$(pick_gpu "${2:-}")"

  local embed_dim
  case "$embed_choice" in
    64|128)
      embed_dim="$embed_choice"
      ;;
    *)
      echo "Unsupported model size: $embed_choice (expected 64 or 128)" >&2
      exit 1
      ;;
  esac

  local train_batch_size="${TRAIN_BATCH_SIZE:-32}"
  local val_batch_size="${VAL_BATCH_SIZE:-32}"
  local num_workers="${NUM_WORKERS:-8}"
  local limit_train_batches="${LIMIT_TRAIN_BATCHES:-1.0}"
  local limit_val_batches="${LIMIT_VAL_BATCHES:-1.0}"
  local max_epochs="${MAX_EPOCHS:-64}"
  local seed="${SEED:-2022}"
  local ckpt_path="${CKPT_PATH:-}"
  local init_ckpt_path="${INIT_CKPT_PATH:-}"
  local run_version="${RUN_VERSION:-reliability_$(date +%Y%m%d_%H%M%S)_dim${embed_dim}_gpu${gpu_id}}"
  local experiment_name="${EXPERIMENT_NAME:-hivt_reliability}"
  local mode_target_policy="${MODE_TARGET_POLICY:-fde_only}"
  local scene_target_policy="${SCENE_TARGET_POLICY:-target_best_mode_fail}"
  local risk_fde_threshold="${RISK_FDE_THRESHOLD:-2.0}"
  local risk_miss_threshold="${RISK_MISS_THRESHOLD:-4.0}"
  local risk_conflict_threshold="${RISK_CONFLICT_THRESHOLD:-1.0}"
  local risk_conflict_min_frames="${RISK_CONFLICT_MIN_FRAMES:-2}"
  local risk_conflict_scope="${RISK_CONFLICT_SCOPE:-target_to_neighbors}"
  local risk_scene_rate_threshold="${RISK_SCENE_RATE_THRESHOLD:-0.5}"
  local reliability_rerank_alpha="${RELIABILITY_RERANK_ALPHA:-0.0}"
  local scene_loss_weight="${SCENE_LOSS_WEIGHT:-0.2}"
  local rank_loss_weight="${RANK_LOSS_WEIGHT:-0.0}"
  local calib_loss_weight="${CALIB_LOSS_WEIGHT:-0.0}"
  local mode_risk_threshold_weight_enabled="${MODE_RISK_THRESHOLD_WEIGHT_ENABLED:-false}"
  local mode_risk_threshold_weight_radius="${MODE_RISK_THRESHOLD_WEIGHT_RADIUS:-0.25}"
  local mode_risk_threshold_weight_peak="${MODE_RISK_THRESHOLD_WEIGHT_PEAK:-2.0}"
  local mode_risk_threshold_weight_base="${MODE_RISK_THRESHOLD_WEIGHT_BASE:-1.0}"
  local mode_risk_rank_top_k="${MODE_RISK_RANK_TOP_K:-0}"
  local mode_risk_rank_near_threshold_only="${MODE_RISK_RANK_NEAR_THRESHOLD_ONLY:-false}"
  local mode_risk_rank_threshold_radius="${MODE_RISK_RANK_THRESHOLD_RADIUS:-0.25}"
  local freeze_backbone="${FREEZE_BACKBONE:-false}"

  check_data_ready
  check_preprocessed_hint
  activate_env
  ensure_dir "$RUNS_ROOT"
  ensure_dir "$LOG_ROOT"

  cd "$PROJECT_ROOT"
  echo "Using GPU $gpu_id for reliability training"
  echo "embed_dim=$embed_dim train_batch_size=$train_batch_size val_batch_size=$val_batch_size num_workers=$num_workers"
  echo "limit_train_batches=$limit_train_batches limit_val_batches=$limit_val_batches"
  echo "max_epochs=$max_epochs seed=$seed ckpt_path=${ckpt_path:-<none>} init_ckpt_path=${init_ckpt_path:-<none>}"
  echo "experiment_name=$experiment_name run_version=$run_version"
  echo "mode_target_policy=$mode_target_policy scene_target_policy=$scene_target_policy conflict_scope=$risk_conflict_scope freeze_backbone=$freeze_backbone"
  echo "rank_loss_weight=$rank_loss_weight calib_loss_weight=$calib_loss_weight threshold_weight=$mode_risk_threshold_weight_enabled top_k=$mode_risk_rank_top_k"

  exec env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python train.py \
    --root "$DATA_ROOT" \
    --embed_dim "$embed_dim" \
    --gpus 1 \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --num_workers "$num_workers" \
    --max_epochs "$max_epochs" \
    --seed "$seed" \
    --limit_train_batches "$limit_train_batches" \
    --limit_val_batches "$limit_val_batches" \
    --experiment_root "$RUNS_ROOT" \
    --experiment_name "$experiment_name" \
    --experiment_version "$run_version" \
    --use_reliability true \
    --reliability_rerank_alpha "$reliability_rerank_alpha" \
    --scene_loss_weight "$scene_loss_weight" \
    --rank_loss_weight "$rank_loss_weight" \
    --calib_loss_weight "$calib_loss_weight" \
    --mode_risk_threshold_weight_enabled "$mode_risk_threshold_weight_enabled" \
    --mode_risk_threshold_weight_radius "$mode_risk_threshold_weight_radius" \
    --mode_risk_threshold_weight_peak "$mode_risk_threshold_weight_peak" \
    --mode_risk_threshold_weight_base "$mode_risk_threshold_weight_base" \
    --mode_risk_rank_top_k "$mode_risk_rank_top_k" \
    --mode_risk_rank_near_threshold_only "$mode_risk_rank_near_threshold_only" \
    --mode_risk_rank_threshold_radius "$mode_risk_rank_threshold_radius" \
    --risk_fde_threshold "$risk_fde_threshold" \
    --risk_miss_threshold "$risk_miss_threshold" \
    --risk_conflict_threshold "$risk_conflict_threshold" \
    --risk_conflict_min_frames "$risk_conflict_min_frames" \
    --risk_conflict_scope "$risk_conflict_scope" \
    --risk_scene_rate_threshold "$risk_scene_rate_threshold" \
    --mode_target_policy "$mode_target_policy" \
    --scene_target_policy "$scene_target_policy" \
    --freeze_backbone "$freeze_backbone" \
    ${init_ckpt_path:+--init_ckpt_path "$init_ckpt_path"} \
    ${ckpt_path:+--ckpt_path "$ckpt_path"}
}

run_train_reliability_shift() {
  local embed_choice="${1:-64}"
  local gpu_id
  gpu_id="$(pick_gpu "${2:-}")"

  local embed_dim
  case "$embed_choice" in
    64|128) embed_dim="$embed_choice" ;;
    *) echo "Unsupported model size: $embed_choice" >&2; exit 1 ;;
  esac

  local train_batch_size="${TRAIN_BATCH_SIZE:-32}"
  local val_batch_size="${VAL_BATCH_SIZE:-32}"
  local num_workers="${NUM_WORKERS:-8}"
  local limit_train_batches="${LIMIT_TRAIN_BATCHES:-1.0}"
  local limit_val_batches="${LIMIT_VAL_BATCHES:-1.0}"
  local max_epochs="${MAX_EPOCHS:-64}"
  local seed="${SEED:-2022}"
  local ckpt_path="${CKPT_PATH:-}"
  local init_ckpt_path="${INIT_CKPT_PATH:-}"
  local run_version="${RUN_VERSION:-reliability_shift_$(date +%Y%m%d_%H%M%S)_dim${embed_dim}_gpu${gpu_id}}"
  local experiment_name="${EXPERIMENT_NAME:-hivt_reliability}"
  local mode_target_policy="${MODE_TARGET_POLICY:-fde_only}"
  local scene_target_policy="${SCENE_TARGET_POLICY:-target_best_mode_fail}"
  local risk_fde_threshold="${RISK_FDE_THRESHOLD:-2.0}"
  local risk_miss_threshold="${RISK_MISS_THRESHOLD:-4.0}"
  local risk_conflict_threshold="${RISK_CONFLICT_THRESHOLD:-1.0}"
  local risk_conflict_min_frames="${RISK_CONFLICT_MIN_FRAMES:-2}"
  local risk_conflict_scope="${RISK_CONFLICT_SCOPE:-target_to_neighbors}"
  local risk_scene_rate_threshold="${RISK_SCENE_RATE_THRESHOLD:-0.5}"
  local reliability_rerank_alpha="${RELIABILITY_RERANK_ALPHA:-0.0}"
  local scene_loss_weight="${SCENE_LOSS_WEIGHT:-0.2}"
  local rank_loss_weight="${RANK_LOSS_WEIGHT:-0.0}"
  local calib_loss_weight="${CALIB_LOSS_WEIGHT:-0.0}"
  local mode_risk_threshold_weight_enabled="${MODE_RISK_THRESHOLD_WEIGHT_ENABLED:-false}"
  local mode_risk_threshold_weight_radius="${MODE_RISK_THRESHOLD_WEIGHT_RADIUS:-0.25}"
  local mode_risk_threshold_weight_peak="${MODE_RISK_THRESHOLD_WEIGHT_PEAK:-2.0}"
  local mode_risk_threshold_weight_base="${MODE_RISK_THRESHOLD_WEIGHT_BASE:-1.0}"
  local mode_risk_rank_top_k="${MODE_RISK_RANK_TOP_K:-0}"
  local mode_risk_rank_near_threshold_only="${MODE_RISK_RANK_NEAR_THRESHOLD_ONLY:-false}"
  local mode_risk_rank_threshold_radius="${MODE_RISK_RANK_THRESHOLD_RADIUS:-0.25}"
  local freeze_backbone="${FREEZE_BACKBONE:-false}"

  check_data_ready
  check_preprocessed_hint
  activate_env
  ensure_dir "$RUNS_ROOT"
  ensure_dir "$LOG_ROOT"

  cd "$PROJECT_ROOT"
  echo "Using GPU $gpu_id for reliability+shift training"
  echo "embed_dim=$embed_dim train_batch_size=$train_batch_size run_version=$run_version"
  echo "limit_train_batches=$limit_train_batches limit_val_batches=$limit_val_batches"
  echo "max_epochs=$max_epochs seed=$seed ckpt_path=${ckpt_path:-<none>} init_ckpt_path=${init_ckpt_path:-<none>}"
  echo "mode_target_policy=$mode_target_policy scene_target_policy=$scene_target_policy conflict_scope=$risk_conflict_scope freeze_backbone=$freeze_backbone"
  echo "rank_loss_weight=$rank_loss_weight calib_loss_weight=$calib_loss_weight threshold_weight=$mode_risk_threshold_weight_enabled top_k=$mode_risk_rank_top_k"

  exec env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python train.py \
    --root "$DATA_ROOT" \
    --embed_dim "$embed_dim" \
    --gpus 1 \
    --train_batch_size "$train_batch_size" \
    --val_batch_size "$val_batch_size" \
    --num_workers "$num_workers" \
    --max_epochs "$max_epochs" \
    --seed "$seed" \
    --limit_train_batches "$limit_train_batches" \
    --limit_val_batches "$limit_val_batches" \
    --experiment_root "$RUNS_ROOT" \
    --experiment_name "$experiment_name" \
    --experiment_version "$run_version" \
    --use_reliability true \
    --reliability_rerank_alpha "$reliability_rerank_alpha" \
    --scene_loss_weight "$scene_loss_weight" \
    --rank_loss_weight "$rank_loss_weight" \
    --calib_loss_weight "$calib_loss_weight" \
    --mode_risk_threshold_weight_enabled "$mode_risk_threshold_weight_enabled" \
    --mode_risk_threshold_weight_radius "$mode_risk_threshold_weight_radius" \
    --mode_risk_threshold_weight_peak "$mode_risk_threshold_weight_peak" \
    --mode_risk_threshold_weight_base "$mode_risk_threshold_weight_base" \
    --mode_risk_rank_top_k "$mode_risk_rank_top_k" \
    --mode_risk_rank_near_threshold_only "$mode_risk_rank_near_threshold_only" \
    --mode_risk_rank_threshold_radius "$mode_risk_rank_threshold_radius" \
    --risk_fde_threshold "$risk_fde_threshold" \
    --risk_miss_threshold "$risk_miss_threshold" \
    --risk_conflict_threshold "$risk_conflict_threshold" \
    --risk_conflict_min_frames "$risk_conflict_min_frames" \
    --risk_conflict_scope "$risk_conflict_scope" \
    --risk_scene_rate_threshold "$risk_scene_rate_threshold" \
    --mode_target_policy "$mode_target_policy" \
    --scene_target_policy "$scene_target_policy" \
    --freeze_backbone "$freeze_backbone" \
    ${init_ckpt_path:+--init_ckpt_path "$init_ckpt_path"} \
    ${ckpt_path:+--ckpt_path "$ckpt_path"} \
    --shift_history_dropout_p "${SHIFT_HISTORY_DROPOUT_P:-0.3}" \
    --shift_neighbor_dropout_p "${SHIFT_NEIGHBOR_DROPOUT_P:-0.2}" \
    --shift_position_noise_std "${SHIFT_POSITION_NOISE_STD:-0.1}" \
    --shift_heading_noise_std "${SHIFT_HEADING_NOISE_STD:-0.05}" \
    --shift_map_jitter_std "${SHIFT_MAP_JITTER_STD:-0.05}" \
    --shift_lane_dropout_p "${SHIFT_LANE_DROPOUT_P:-0.1}"
}

run_eval() {
  local ckpt_path="${1:-}"
  local gpu_id
  gpu_id="$(pick_gpu "${2:-}")"
  local batch_size="${3:-${EVAL_BATCH_SIZE:-32}}"
  local num_workers="${NUM_WORKERS:-8}"

  if [[ -z "$ckpt_path" ]]; then
    echo "Missing checkpoint path for eval" >&2
    usage
    exit 1
  fi

  check_data_ready
  require_file "$ckpt_path"
  activate_env

  cd "$PROJECT_ROOT"
  echo "Using GPU $gpu_id for evaluation"
  echo "ckpt_path=$ckpt_path batch_size=$batch_size num_workers=$num_workers"

  exec env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python eval.py \
    --root "$DATA_ROOT" \
    --ckpt_path "$ckpt_path" \
    --gpus 1 \
    --batch_size "$batch_size" \
    --num_workers "$num_workers"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local mode="$1"
  shift

  case "$mode" in
    train)
      run_train "$@"
      ;;
    train_reliability)
      run_train_reliability "$@"
      ;;
    train_reliability_shift)
      run_train_reliability_shift "$@"
      ;;
    eval)
      run_eval "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
