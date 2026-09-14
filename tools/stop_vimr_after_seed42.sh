#!/usr/bin/env bash
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
session="vimr_coco_full"
seed_dir="${repo_dir}/outputs/vimr_coco_full/seed_42"
metrics_file="${seed_dir}/metrics.json"
instances_file="${seed_dir}/per_instance.csv"
log_file="${repo_dir}/outputs/vimr_coco_full/stop_after_seed42.log"
expected_lines=46251

printf '[watcher] armed: waiting for complete seed 42 outputs\n' >> "${log_file}"

while tmux has-session -t "${session}" 2>/dev/null; do
    if [[ -s "${metrics_file}" && -s "${instances_file}" ]] \
        && awk -v expected="${expected_lines}" 'END { exit !(NR >= expected) }' "${instances_file}"; then
        printf '[watcher] seed 42 outputs complete; stopping the training command\n' >> "${log_file}"
        tmux send-keys -t "${session}" C-c
        sleep 3
        printf '[watcher] stop signal sent; watcher exiting\n' >> "${log_file}"
        exit 0
    fi
    sleep 2
done

printf '[watcher] target tmux session disappeared before seed 42 completed\n' >> "${log_file}"
exit 1
