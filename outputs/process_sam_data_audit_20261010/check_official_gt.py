"""Stream COCO annotations to check collected masks without loading 470 MB at once."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as masks

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
data = ROOT / "outputs/cais_synthetic_click2_click3_300_20260921/alignment/modeling_clicks_strict.jsonl"
records = [json.loads(l) for l in data.read_text(encoding="utf-8").splitlines() if l.strip()]
records = [r for r in records if r.get("source","human")=="human"]
needed = {}
for record in records:
    needed.setdefault(record["annotation_id"],[]).append(record)

def annotations(path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        while True:
            chunk = handle.read(1<<20)
            if not chunk:
                raise ValueError("COCO annotations array not found")
            buffer += chunk
            index = buffer.find('"annotations"')
            if index >= 0:
                index = buffer.find("[",index)
                if index >= 0:
                    buffer,position = buffer[index+1:],0
                    break
            buffer = buffer[-64:]
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value,end = decoder.raw_decode(buffer,position)
            except json.JSONDecodeError:
                chunk = handle.read(1<<20)
                if not chunk:
                    raise
                buffer,position = buffer[position:]+chunk,0
                continue
            position = end
            yield value

found = set()
issues = []
pixel_difference_max = 0
for annotation in annotations(ROOT / "dataset/coco/annotations_trainval2017/annotations/instances_train2017.json"):
    annotation_id = annotation["id"]
    if annotation_id not in needed:
        continue
    found.add(annotation_id)
    for r in needed[annotation_id]:
        if annotation["image_id"] != r["coco_image_id"] or annotation["category_id"] != r["category_id"]:
            issues.append({"event_id":r["event_id"],"issue":"annotation_identity_mismatch"})
        segmentation = annotation["segmentation"]
        if isinstance(segmentation,list):
            rle = masks.merge(masks.frPyObjects(segmentation,r["image_height_px"],r["image_width_px"]))
        elif isinstance(segmentation["counts"],list):
            rle = masks.frPyObjects(segmentation,r["image_height_px"],r["image_width_px"])
        else:
            rle = segmentation
        official = masks.decode(rle).astype(bool)
        collected = masks.decode(r["ground_truth_rle"]).astype(bool)
        difference = int(np.count_nonzero(official != collected))
        pixel_difference_max = max(pixel_difference_max,difference)
        if difference:
            issues.append({"event_id":r["event_id"],"issue":"ground_truth_mask_mismatch","different_pixels":difference})
    if len(found)==len(needed):
        break
report={"unique_annotation_ids":len(needed),"official_annotation_ids_found":len(found),"verified_click_records":sum(len(needed[a]) for a in found),"missing_annotation_ids":sorted(set(needed)-found),"pixel_difference_max":pixel_difference_max,"issues":issues}
(OUT/"official_gt_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False))
