"""Audit the exact saved experiment cohort against local raw collection files."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from efficientvit.models.cais.coco_click_dataset import (
    PROCESS_FEATURE_NAMES, _resample_trajectory_fixed_window,
    is_high_quality_trajectory, process_features,
)
from efficientvit.models.process_sam.dataset import viewport_state
from tools.audit_cais_process_trajectory import trajectory_row, summarize

RUN = ROOT / "outputs/process_sam_e0_e4_full_20261010_seed9527"
OUT = Path(__file__).parent
manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
dataset_root = ROOT / manifest["args"]["dataset_root"]
data_file = dataset_root / "alignment/modeling_clicks_strict.jsonl"
all_records = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
selected_ids = {event_id for ids in manifest["selected_event_ids"].values() for event_id in ids}
records = [r for r in all_records if r["event_id"] in selected_ids]
by_task = defaultdict(list)
for r in records:
    by_task[r["task_run_id"]].append(r)
for rows in by_task.values():
    rows.sort(key=lambda r: r["click_index"])

issues = Counter()
issue_examples = defaultdict(list)
def issue(name, r):
    issues[name] += 1
    if len(issue_examples[name]) < 5:
        issue_examples[name].append(r["event_id"])

quality_rows, feature_rows, valid_samples = [], [], []
quality_by_id, viewport_changes = {}, {}
image_sizes = {}
metric_max_error = defaultdict(float)
prompt_delta_px = []
history_missing = 0
viewport_valid = 0
partial_viewport = 0
point_count, negative_count = 0, 0
for task, rows in by_task.items():
    mapping = {r["click_index"]: r for r in rows}
    if sorted(mapping) != list(range(1, max(mapping) + 1)):
        issue("non_contiguous_click_sequence", rows[-1])
    views = [viewport_state(r) for r in rows]
    zoom_change = any(abs(np.log(views[i][0][0] / views[i-1][0][0])) > 0.01 for i in range(1,len(views)) if views[i][1] and views[i-1][1])
    pan_change = any(np.linalg.norm(np.array(views[i][0][1:3])-views[i-1][0][1:3]) > 0.01 for i in range(1,len(views)) if views[i][1] and views[i-1][1])
    viewport_changes[task] = {"zoom_change_between_clicks":bool(zoom_change), "pan_change_between_clicks":bool(pan_change)}
    for r in rows:
        previous = mapping.get(r["click_index"] - 1)
        if previous and r["click_timestamp_ms"] <= previous["click_timestamp_ms"]:
            issue("non_increasing_click_timestamp", r)
        if len(r["prompt_history"]) != r["click_index"]:
            issue("prompt_history_length_mismatch", r)
        for i,prompt in enumerate(r["prompt_history"],1):
            earlier = mapping.get(i)
            if earlier is None:
                history_missing += 1
                continue
            if earlier["prompt_history"][-1] != prompt:
                issue("prompt_history_changed", r)
            if prompt["click_type"] != earlier["click_type"]:
                issue("prompt_history_label_mismatch", r)
        xy = np.asarray([r["click_x_norm"],r["click_y_norm"]])
        if not np.isfinite(xy).all() or ((xy<0)|(xy>1)).any():
            issue("invalid_click_coordinate",r)
        expected = (r["image_width_px"],r["image_height_px"])
        gt = mask_utils.decode(r["ground_truth_rle"]).astype(bool)
        if gt.ndim == 3:
            gt = gt.any(axis=2)
        if gt.shape != (expected[1],expected[0]) or not gt.any():
            issue("invalid_ground_truth_shape_or_empty",r)
        x,y = (np.floor(xy*np.array(expected))).astype(int)
        x,y = np.clip(x,0,expected[0]-1),np.clip(y,0,expected[1]-1)
        if bool(gt[y,x]) != bool(r["click_type"]):
            issue("click_label_gt_disagreement",r)
        current_prompt = r["prompt_history"][-1]
        prompt_delta_px.append(float(np.linalg.norm((np.array([current_prompt["x_norm"],current_prompt["y_norm"]])-xy)*expected)))
        for key in ["image_ref","pre_mask_ref","post_mask_ref"]:
            path = dataset_root / r[key]
            if not path.is_file():
                issue("missing_"+key,r)
                continue
            if key == "image_ref" and str(path) in image_sizes:
                size = image_sizes[str(path)]
            else:
                with Image.open(path) as im:
                    size = im.size
                    if key != "image_ref":
                        mask = np.asarray(im.convert("L")) > 0
                if key == "image_ref":
                    image_sizes[str(path)] = size
            if size != expected:
                issue("size_mismatch_"+key,r)
            elif key != "image_ref":
                union = (mask|gt).sum()
                actual_iou = float((mask&gt).sum()/union) if union else 1.0
                metrics_key = key.replace("_ref","_metrics")
                stored_iou = r.get(metrics_key,{}).get("iou")
                if stored_iou is not None:
                    error = abs(actual_iou-stored_iou)
                    metric_max_error[key] = max(metric_max_error[key],error)
                    if error > 1e-5:
                        issue("stored_iou_mismatch_"+key,r)
        state, good = viewport_state(r)
        viewport_valid += int(good)
        if good and (state[3] < 0.98 or state[4] < 0.98):
            partial_viewport += 1
        trajectory = r.get("preceding_trajectory",{}).get("points",[])
        points = np.asarray(trajectory,dtype=float)
        if points.size and (not np.isfinite(points).all() or (np.diff(points[:,0])<0).any() or (points[:,0]>1e-6).any()):
            issue("invalid_preclick_trajectory",r)
        quality = trajectory_row(r,previous)
        quality_rows.append(quality)
        quality_by_id[r["event_id"]] = is_high_quality_trajectory(r,previous)
        feature_rows.append(process_features(r,previous))
        _,sample_valid = _resample_trajectory_fixed_window(trajectory,*xy,16,2000)
        valid_samples.append(int(sample_valid.sum()))
        point_count += len(trajectory)
        negative_count += int(r["click_type"]==0)
    if len(quality_rows)%200 < len(rows):
        print(f"Audited {len(quality_rows)}/{len(records)} recorded clicks",flush=True)

raw_stats = {}
raw_models = Counter()
linked_raw = {}
linked_summaries = {}
for name in sorted({r["human_dataset"] for r in records}):
    source = Path(next(r["image_ref"] for r in records if r["human_dataset"]==name)).parents[2] / "interaction_data"
    def load(name):
        return [json.loads(l) for l in (source/name).read_text(encoding="utf-8").splitlines() if l.strip()]
    raw = load("point_interactions.jsonl")
    summaries = load("interaction_tasks.jsonl")
    events = load("interaction_task_events.jsonl")
    raw_map = {r["event_id"]:r for r in raw}
    summary_map = {r["task_run_id"]:r for r in summaries}
    selected = [r for r in records if r["human_dataset"]==name]
    selected_tasks = {r["task_run_id"] for r in selected}
    selected_summaries = [summary_map[t] for t in selected_tasks if t in summary_map]
    linked_raw.update(raw_map)
    linked_summaries.update(summary_map)
    real_zoom = lambda a: a.get("type")=="zoom" and abs(float(a.get("viewport",{}).get("delta_log_zoom",0)))>0.01
    real_pan = lambda a: a.get("type")=="pan" and np.linalg.norm([a.get("viewport",{}).get("delta_center_x_norm",0),a.get("viewport",{}).get("delta_center_y_norm",0)])>0.01
    full_point_counts = [len(r.get("full_trajectory",{}).get("points",[])) for r in selected_summaries]
    raw_stats[name] = {
        "raw_point_records":len(raw),"raw_tasks":len(summaries),"raw_termination_counts":dict(Counter(r["termination"] for r in summaries)),
        "selected_clicks":len(selected),"selected_tasks":len(selected_tasks),
        "selected_missing_raw_event_ids":sum(r["event_id"] not in raw_map for r in selected),
        "selected_missing_task_summaries":len(selected_tasks)-len(selected_summaries),
        "selected_termination_counts":dict(Counter(r["termination"] for r in selected_summaries)),
        "selected_full_trajectory_nonempty_tasks":sum(n>0 for n in full_point_counts),
        "selected_full_trajectory_points_median":float(np.median(full_point_counts)),
        "selected_action_counts":dict(Counter(a.get("type") for r in selected_summaries for a in r["actions"])),
        "selected_tasks_with_nontrivial_zoom_actions":sum(any(real_zoom(a) for a in r["actions"]) for r in selected_summaries),
        "selected_tasks_with_nontrivial_pan_actions":sum(any(real_pan(a) for a in r["actions"]) for r in selected_summaries),
        "raw_start_finish_counts":dict(Counter(e["action_type"] for e in events)),
    }
    for r in selected:
        if r["event_id"] in raw_map:
            original = raw_map[r["event_id"]]
            raw_models[original["interaction_context"].get("model_file","missing")] += 1
            if original["preceding_trajectory"] != r["preceding_trajectory"]:
                issue("trajectory_changed_during_export",r)

groups = {}
for split in ["train","val","test"]:
    rows = [r for r in records if r["split"]==split]
    tasks = {r["task_run_id"] for r in rows}
    groups[split] = {
        "clicks":len(rows),"tasks":len(tasks),"images":len({r["coco_image_id"] for r in rows}),
        "click_index_counts":dict(Counter(r["click_index"] for r in rows)),
        "complete_five_click_tasks":sum(len(by_task[t])==5 for t in tasks),
        "unique_annotators":len({r["annotator_id"] for r in rows}),
        "categories":len({r["category_id"] for r in rows}),
        "quality_eligible_clicks":sum(quality_by_id[r["event_id"]] for r in rows),
        "tasks_with_zoom_change_between_clicks":sum(viewport_changes[t]["zoom_change_between_clicks"] for t in tasks),
        "tasks_with_pan_change_between_clicks":sum(viewport_changes[t]["pan_change_between_clicks"] for t in tasks),
    }
image_sets={s:{r["coco_image_id"] for r in records if r["split"]==s} for s in groups}
leakage={f"{a}_{b}":len(image_sets[a]&image_sets[b]) for a,b in [("train","val"),("train","test"),("val","test")]}

def paired_delta(left,right):
    assert {r["event_id"] for r in left}=={r["event_id"] for r in right}
    right_map={r["event_id"]:r["iou"] for r in right}
    task_deltas=defaultdict(list)
    for r in left:
        task_deltas[r["task_run_id"]].append(r["iou"]-right_map[r["event_id"]])
    values=np.array([np.mean(v) for v in task_deltas.values()])
    generator=np.random.default_rng(9527)
    boot=values[generator.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return {"tasks":len(values),"mean_task_delta":float(values.mean()),"bootstrap_95ci":np.quantile(boot,[.025,.975]).tolist(),"mean_event_delta":float(np.mean([r["iou"]-right_map[r["event_id"]] for r in left]))}
evaluations={name:json.loads((RUN/name/"test_events.json").read_text(encoding="utf-8")) for name in ["E0","E1","E2","E3","E4"]}
paired = {f"{name}_minus_E0":paired_delta(evaluations[name],evaluations["E0"]) for name in ["E1","E2","E3"]}
strata = {}
for label in ["quality_eligible","other"]:
    ids={r["event_id"] for r in records if r["split"]=="test" and quality_by_id[r["event_id"]]==(label=="quality_eligible")}
    subset = {name:[r for r in rows if r["event_id"] in ids] for name,rows in evaluations.items()}
    strata[label]={"events":len(ids),"mean_event_iou":{name:float(np.mean([r["iou"] for r in rows])) for name,rows in subset.items()},"E3_minus_E0":paired_delta(subset["E3"],subset["E0"])}
matrix=np.stack(feature_rows)
history={name:[{"epoch":h["epoch"],"loss":h["train_loss"],"val_iou":h["val"]["mean_iou"]} for h in json.loads((RUN/name/"history.json").read_text(encoding="utf-8"))] for name in ["E0","E1","E2","E3"]}
report={
    "run":str(RUN),"dataset_file":str(data_file),"dataset_hash_matches_training":hashlib.sha256(data_file.read_bytes()).hexdigest()==manifest["dataset_sha256"],
    "all_file_sources":dict(Counter(r.get("source","human") for r in all_records)),"selected_sources":dict(Counter(r.get("source","human") for r in records)),
    "selected_id_count":len(selected_ids),"selected_record_count":len(records),"duplicate_event_ids":len(records)-len({r["event_id"] for r in records}),
    "asset_and_alignment_issues":dict(issues),"issue_examples":dict(issue_examples),"stored_mask_iou_max_error":dict(metric_max_error),
    "missing_history_slots":history_missing,"prompt_coordinate_quantization_error_px":{"median":float(np.median(prompt_delta_px)),"max":float(max(prompt_delta_px))},
    "splits":groups,"image_split_overlap":leakage,"raw_datasets":raw_stats,"collection_model_files":dict(raw_models),"training_base_checkpoint":manifest["args"]["base_checkpoint"],
    "viewport_valid_clicks":viewport_valid,"viewport_partially_excludes_image_clicks":partial_viewport,"negative_clicks":negative_count,
    "trajectory_quality_heuristic":summarize(quality_rows),"quality_is_heuristic_not_proof_of_data_loss":True,
    "preclick_trajectory_point_count":point_count,"fixed_window_valid_samples_median":float(np.median(valid_samples)),"fixed_window_zero_valid_events":sum(n==0 for n in valid_samples),
    "process_feature_std":dict(zip(PROCESS_FEATURE_NAMES,matrix.std(axis=0).astype(float).tolist())),
    "paired_test_comparisons":paired,"test_quality_strata":strata,"training_history":history,
    "adapter_omissions":["task-level full_trajectory","timestamped zoom/pan actions between clicks","task-level encoder wait / reset / preview events"],
    "limitations":["raw full trajectories are sampled motion observations, not continuous time coverage","quality thresholds do not establish causal reasons for performance","single training seed; evaluation has only 12 complete five-click tasks","recorded L1 pre-masks / click policy are replayed into L0, not a closed-loop L0 interaction run"]
}
(OUT/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ["training_history","issue_examples","process_feature_std"]},ensure_ascii=False,indent=2,allow_nan=False))
