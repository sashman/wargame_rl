"""Latest logged metrics, read from the local .wandb datastore (no API call)."""

import glob
import sys

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

WANT = ("coherency_rate", "models_out_of_coherency", "vp_margin", "fraction_alive")

for path in sorted(glob.glob(sys.argv[1])):
    ds = DataStore()
    try:
        ds.open_for_scan(path)
    except Exception:
        continue
    latest = {}
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        for item in rec.history.item:
            name = item.key or "/".join(item.nested_key)
            if name.startswith("eval/") and name.split("/")[-1] in WANT:
                try:
                    latest[name.split("/")[-1]] = float(item.value_json)
                except ValueError:
                    pass
    print({k: round(v, 3) for k, v in sorted(latest.items())})
