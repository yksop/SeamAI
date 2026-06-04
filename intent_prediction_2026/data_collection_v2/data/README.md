# data/

Default output folder for `../data_collection/pose_data_collection.py`. The
recorder creates one dated folder per day and **appends** same-day sessions:

```
data/
  2026-05-29/
    enter/    <clip>.json  <clip>.mp4  <clip>_pose.json
    pass/
    exit/
    removed/  too_short/  too_few_points/  partial_bottom/  corrupt/
    _sessions/   session_<date>_<time>.json   (per-session metadata + counts)
    _temp/       scratch, emptied as clips finalize
```

File name: `YYYY-MM-DD_HHMMSS_NNNN_id<track>_<label>`. Nothing is deleted —
filtered clips are sorted into `removed/<reason>/`. See
`../data_collection/README.md` for the JSON schema.

(This folder is intentionally kept in version control via this README; the
recordings themselves are data, not code.)
