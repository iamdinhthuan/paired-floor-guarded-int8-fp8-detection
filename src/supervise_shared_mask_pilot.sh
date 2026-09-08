#!/usr/bin/env bash
set -uo pipefail

pilot_root=${1:-/home/thuan/topic_c_ivc}
pilot_root=$(realpath -e -- "$pilot_root") || {
  printf 'shared-mask project root is absent\n' >&2
  exit 66
}
attempt=${2:-shared_mask_pilot_v2}
config_arg=${3:-configs/${attempt}.json}
if [[ ! "$attempt" =~ ^shared_mask_pilot_v[0-9]+$ ]]; then
  printf 'invalid shared-mask attempt namespace: %s\n' "$attempt" >&2
  exit 64
fi
if [[ "$config_arg" = /* ]]; then
  config_path=$config_arg
else
  config_path="$pilot_root/$config_arg"
fi
config_path=$(realpath -e -- "$config_path") || {
  printf 'shared-mask config is absent: %s\n' "$config_arg" >&2
  exit 66
}
case "$config_path" in
  "$pilot_root"/*) ;;
  *)
    printf 'shared-mask config escapes project root: %s\n' "$config_path" >&2
    exit 64
    ;;
esac

log_dir="$pilot_root/outputs/logs/$attempt"
report_dir="$pilot_root/outputs/reports/$attempt"
driver_pid_file="$log_dir/driver.pid"
supervisor_pid_file="$log_dir/supervisor.pid"
driver_log="$log_dir/driver.log"
supervisor_log="$log_dir/supervisor.log"
progress_file="$report_dir/progress.json"
completion_file="$report_dir/complete.json"
completion_marker="$completion_file.complete"

mkdir -p "$log_dir"
exec 9>"$log_dir/supervisor.lock"
if ! flock -n 9; then
  printf '%s supervisor already active\n' "$(date --iso-8601=seconds)" >>"$supervisor_log"
  exit 0
fi
printf '%s\n' "$$" >"$supervisor_pid_file"

cleanup_supervisor_pid() {
  if [[ -f "$supervisor_pid_file" ]] && [[ $(tr -d '[:space:]' <"$supervisor_pid_file") == "$$" ]]; then
    rm -f -- "$supervisor_pid_file"
  fi
}
trap cleanup_supervisor_pid EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source /home/thuan/miniconda3/etc/profile.d/conda.sh || exit 69
conda activate qtsd || exit 69
export PYTHONPATH="$pilot_root/src"
if [[ "$attempt" == "shared_mask_pilot_v3" ]]; then
  export SHARED_MASK_PILOT_ATTEMPT="$attempt"
  driver_script="$pilot_root/src/run_shared_mask_pilot_v3.py"
else
  driver_script="$pilot_root/src/run_shared_mask_pilot.py"
fi

log_event() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$supervisor_log"
}

driver_is_alive() {
  [[ -f "$driver_pid_file" ]] || return 1
  local driver_pid
  driver_pid=$(tr -d '[:space:]' <"$driver_pid_file")
  [[ "$driver_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$driver_pid" 2>/dev/null || return 1
  [[ -r "/proc/$driver_pid/cmdline" ]] || return 1
  local command_line
  command_line=$(tr '\0' ' ' <"/proc/$driver_pid/cmdline")
  [[ "$command_line" == *"$driver_script"* ]] \
    && [[ "$command_line" == *"$config_path"* ]]
}

completion_is_valid() {
  [[ -f "$completion_file" && -f "$completion_marker" ]] || return 1
  python - "$completion_file" "$completion_marker" "$config_path" "$attempt" "$pilot_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

report_path, marker_path, config_path, attempt, root_value = sys.argv[1:]
try:
    report = json.load(open(report_path, encoding="utf-8"))
    config = json.load(open(config_path, encoding="utf-8"))
    marker = open(marker_path, encoding="utf-8").read().strip()
except (OSError, ValueError):
    raise SystemExit(1)

def canonical_hash(document, field):
    payload = {key: value for key, value in document.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

root = Path(root_value).resolve()
artifacts = report.get("artifacts_sha256")
artifacts_valid = isinstance(artifacts, dict) and bool(artifacts)
if artifacts_valid:
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            artifacts_valid = False
            break
        candidate = Path(relative)
        if candidate.is_absolute():
            artifacts_valid = False
            break
        resolved = (root / candidate).resolve()
        if root != resolved and root not in resolved.parents:
            artifacts_valid = False
            break
        if not resolved.is_file() or file_hash(resolved) != expected:
            artifacts_valid = False
            break

declared = report.get("report_sha256")
valid = (
    report.get("attempt") == attempt
    and report.get("status") == "complete"
    and isinstance(declared, str)
    and declared == canonical_hash(report, "report_sha256")
    and marker == declared
    and report.get("config_sha256") == config.get("config_sha256")
    and config.get("attempt") == attempt
    and config.get("config_sha256") == canonical_hash(config, "config_sha256")
    and artifacts_valid
)
if attempt == "shared_mask_pilot_v3":
    valid = valid and (
        report.get("runtime_evidence_status") == "inadmissible"
        and report.get("fp8_rebuilt_or_rerun") is False
        and report.get("reused_default_fp8_arms") == 39
        and report.get("shared_clean_arms") == 3
        and report.get("shared_corrupted_arms") == 36
        and report.get("direct_cells") == 36
        and report.get("execution_policy") == config.get("execution")
    )
raise SystemExit(0 if valid else 1)
PY
}

transient_gpu_block() {
  [[ -f "$progress_file" ]] || return 1
  python - "$progress_file" <<'PY'
import json
import sys

try:
    document = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
message = str(document.get("error", ""))
prefixes = (
    "GPU idle gate found another compute process:",
    "GPU capacity gate found ",
    "GPU capacity gate could not query free memory",
)
raise SystemExit(0 if document.get("stage") == "blocked" and message.startswith(prefixes) else 1)
PY
}

gpu_is_available() {
  if [[ "$attempt" == "shared_mask_pilot_v3" ]]; then
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | head -n 1 | tr -d '[:space:]')
    [[ "$free_mib" =~ ^[0-9]+$ ]] || return 1
    # V3 permits unrelated resident services. The driver independently
    # rechecks phase-specific reservations; 12 GiB admits its serial build.
    (( free_mib >= 12288 ))
    return
  fi
  local active
  active=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sed 's#^.*/##' \
    | grep -vx 'sunshine' || true)
  [[ -z "$active" ]]
}

pilot_child_is_alive() {
  python - "$attempt" "$$" <<'PY'
from pathlib import Path
import os
import sys

attempt, supervisor_pid = sys.argv[1], int(sys.argv[2])
workers = {
    "quantize_yolo_onnx.py",
    "build_yolo_trt_engine.py",
    "coco_infer_trt.py",
    "cross_family_infer_trt.py",
    "coco_eval.py",
    "analyze_shared_mask_pilot.py",
}
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in {os.getpid(), supervisor_pid}:
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        continue
    if attempt in command and any(name in command for name in workers):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

log_event "supervisor started pid=$$"
retry_count=0
max_driver_retries=8
while true; do
  if completion_is_valid; then
    log_event "validated completion report present; supervisor exiting"
    exit 0
  fi
  if [[ -e "$completion_file" || -e "$completion_marker" ]]; then
    log_event "invalid or incomplete completion artifact; supervisor exiting"
    exit 3
  fi
  if driver_is_alive; then
    sleep 30
    continue
  fi
  if [[ -f "$progress_file" ]] && ! transient_gpu_block; then
    stage=$(python - "$progress_file" <<'PY'
import json
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
    print(value.get("stage", "unknown"))
except (OSError, ValueError):
    print("invalid-progress")
PY
)
    if [[ "$stage" == "blocked" ]]; then
      log_event "driver stopped at explicit non-transient blocked stage; supervisor exiting"
      exit 2
    fi
    if pilot_child_is_alive; then
      log_event "driver absent but V3 child still active at stage=$stage; waiting"
      sleep 30
      continue
    fi
    if (( retry_count >= max_driver_retries )); then
      log_event "driver retry budget exhausted at stage=$stage; supervisor exiting"
      exit 2
    fi
    retry_count=$((retry_count + 1))
    backoff=$((5 * (1 << (retry_count - 1))))
    if (( backoff > 300 )); then backoff=300; fi
    log_event "driver ended unexpectedly at stage=$stage; retry=$retry_count/$max_driver_retries backoff=${backoff}s"
    sleep "$backoff"
  fi
  if ! gpu_is_available; then
    log_event "GPU occupied; waiting before resume"
    sleep 30
    continue
  fi

  log_event "GPU available; resuming immutable pilot driver"
  python -u "$driver_script" \
    --project-root "$pilot_root" \
    --config "$config_path" \
    --trt-root /home/thuan/traffic/third_party/TensorRT-11.1.0.106 \
    --bootstrap-workers 3 >>"$driver_log" 2>&1 &
  driver_pid=$!
  printf '%s\n' "$driver_pid" >"$driver_pid_file"
  if wait "$driver_pid"; then
    driver_status=0
  else
    driver_status=$?
  fi
  log_event "driver exited status=$driver_status pid=$driver_pid"
  if [[ -f "$driver_pid_file" ]] && [[ $(tr -d '[:space:]' <"$driver_pid_file") == "$driver_pid" ]]; then
    rm -f -- "$driver_pid_file"
  fi
  sleep 5
done
