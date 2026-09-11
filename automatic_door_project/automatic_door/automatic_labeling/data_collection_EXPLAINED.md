# data_collection.py — explained, and a review

This document explains what `data_collection.py` (original by Hanna Norberg)
does, points out the weak spots, and describes `data_collection_improved.py`, a
refactored version that produces the **same output** and can be used
interchangeably in the cleaning pipeline.

---

## 1. What the script is for

It records a training dataset for the smart-door project. A camera watches the
area in front of a door; the script detects and tracks each person, saves a
short video clip per person, and automatically labels each clip as **enter**
(the person walked into the door zone) or **pass** (they walked by). Each clip
also gets a JSON file with the person's trajectory (their position in every
frame). That enter/pass dataset is what the rest of the pipeline trains on.

## 2. The big picture

For every camera frame the script does the same loop:

1. **Read** a frame from the camera and undistort it (lens correction).
2. **Detect + track** people with YOLO11 + the BoT-SORT tracker. Tracking gives
   each person a stable `id` that persists across frames.
3. For each tracked person, **append** their bounding box and center point to
   that person's trajectory, and **write** the current frame into that person's
   video.
4. For people who were being tracked but are **not** seen this frame, count how
   many frames they've been missing. Once they've been gone long enough, **close
   and label** their clip.
5. **Show** a live preview window with boxes, ids, the door zone and an FPS
   readout. Press `q` to stop.

So the unit of output is "one person, from the moment they appear until a short
while after they leave."

## 3. Key concepts

**Track id and the recording lifecycle.** The first time an id appears,
`start_recording` opens a temporary `.mp4` writer and an empty trajectory list.
While the id is seen, frames and positions are appended. When the id has been
missing long enough, `finish_recording` closes the clip.

**The "vanish" buffer.** A tracker briefly loses people (occlusion, a missed
detection). `vanish_threshold` is how many missing frames are tolerated before a
person is considered gone, and `extra_seconds` keeps recording a little longer
so the clip captures them fully leaving. While a person is missing, the script
still logs an *empty* trajectory frame (`center: null, bbox: null`) and keeps
writing video — so a clip has a short "empty tail" at the end.

**enter vs pass — the door ellipse.** The door is modeled as an ellipse anchored
at the bottom-center of the frame (`is_inside_ellipse`). When a clip is closed,
the script looks at the **last frame where the person was actually detected**.
If that last position is inside the ellipse, the clip is **enter**; otherwise
**pass**. `ellipse_axes` sets how big that door zone is.

**Clean frame vs display frame.** Two copies of each frame exist: a *clean* one
that is written to disk (no boxes drawn on it) and a *display* one with boxes,
ids, the ellipse and FPS drawn on top, shown only in the preview window. This is
a nice detail — the saved videos stay free of overlays.

## 4. Output format (what the pipeline consumes)

```
live_output/                         (live_output_test/ if TEST_REC = True)
  enter/   <timestamp>_id<ID>_enter.mp4   + <timestamp>_id<ID>_enter.json
  pass/    <timestamp>_id<ID>_pass.mp4    + <timestamp>_id<ID>_pass.json
  temp/    clips still being recorded
```

Each JSON:

```json
{
  "id": 7,
  "label": "enter",
  "frames": [
    {"frame_index": 0, "center": [620, 410], "bbox": [560, 200, 690, 620]},
    {"frame_index": 1, "center": [618, 405], "bbox": [558, 198, 688, 615]},
    {"frame_index": 2, "center": null, "bbox": null}
  ]
}
```

`center` is the midpoint of `bbox`; both are `null` on frames where the person
was missing.

## 5. Settings that matter

| Setting | What it controls |
|---|---|
| `ellipse_axes` | size of the door zone that decides enter vs pass |
| `vanish_threshold` | missing frames tolerated before a clip is closed |
| `extra_seconds` | extra footage kept after the threshold is hit |
| `camera_index` | which camera to read (`0` = `/dev/video0`) |
| `computer` | which lens calibration to apply (`"Jetson"` / `"ASUS"`) |
| `distort` | whether to undistort + crop to the valid region |
| `test_rec` | write to `live_output_test/` for a throwaway run |

---

## 6. Review — what's weak or risky

Nothing here is catastrophic; the structure is clean and readable. But a few
things are worth knowing.

**1. FPS is overloaded and mutated (the main issue).** One variable, `self.fps`,
is used for three different things: the saved video's frame rate, the
frames↔seconds math (the "too short" check and the vanish buffer), and the live
on-screen counter. The main loop overwrites it every second with the *measured*
display rate. Consequences:

- The video writer is created with whatever `self.fps` happens to be when a clip
  starts, so different clips can be saved with different declared frame rates,
  and the saved videos play back too fast or too slow relative to real time.
- The "too short" threshold (`< self.fps`) and the vanish buffer
  (`self.fps * extra_seconds`) change meaning during a session as the measured
  rate drifts.

**2. No camera-open check.** If the camera can't be opened, `cap.read()` simply
returns nothing and the program exits silently with no explanation.

**3. Cleanup isn't guaranteed.** Cleanup only runs after the loop exits via `q`.
A `Ctrl-C` (or an exception) skips it, leaving the camera and video writers
unreleased and temp clips possibly corrupt.

**4. In-progress clips are lost on quit.** Pressing `q` while people are still
being tracked abandons their temp clips (no label, no JSON). They sit in
`temp/`, unused. Harmless to the dataset, but data you recorded is silently
dropped.

**5. Small things.** `import math`, `from collections import defaultdict` and
`self.frame_rate` are unused. `if test_rec == True: ... elif test_rec == False:`
is non-pythonic and would crash if `test_rec` were ever `None`. In `debug` mode,
appending `tuple(last_center)` would crash if a clip never had a valid detection
(in practice it always has at least one, so it doesn't bite — but it's latent).

**6. The trajectory has no real timestamps.** Frames are indexed `0, 1, 2, …`
with no recorded time, so converting frames to seconds downstream relies on a
fixed assumed FPS. If the capture rate isn't steady, that time axis is only
approximate. (Left as-is to keep the format identical — see below.)

---

## 7. What `data_collection_improved.py` changes

It keeps the detection, tracking, labeling rule and **file format identical**,
so it's a drop-in replacement. The changes are:

- **Fixed timeline FPS.** `output_fps` is decided once (from the camera, or set
  explicitly) and used for both the video writer and the timing math. The
  on-screen counter is a separate, cosmetic value. This removes the drift and
  the wrong-playback-speed problem (issue 1).
- **Camera-open check** with a clear error message (issue 2).
- **`try/finally`** around the loop so the camera and writers are always
  released, even on `Ctrl-C` (issue 3).
- **Cleaner code**: dead imports/variables removed, settings gathered in one
  block at the top, the calibration stored as plain data, short docstrings, and
  the long `run()` split into small helpers (`_undistort`, `_draw_box`,
  `_draw_overlays`, `_update_display_fps`). The latent `debug` crash is guarded
  (issue 5).

**Deliberately kept the same** (so your scripts keep working): the output folder
layout, the `<timestamp>_id<ID>_<label>.mp4` / `.json` naming, the JSON schema
(`id`, `label`, `frames` with `frame_index` / `center` / `bbox`), the enter/pass
ellipse rule, the "skip clips shorter than ~1 s" rule, the empty-tail frames
during vanish, and the YOLO/BoT-SORT settings.

**One intended behavior difference:** because the FPS is now fixed instead of
drifting, the exact "too short" cutoff and vanish buffer are consistent for the
whole session. On a borderline clip this could flip a keep/drop decision
compared to the original — but that is the bug fix, not a format change, and the
dataset it writes is the same shape either way.

**Not changed (would break the format):** adding real per-frame timestamps
(issue 6). If you ever want a more accurate time axis, the clean way is to add a
top-level `fps` field or a per-frame `t` to the JSON and teach the cleaning step
to read it — a deliberate format change to make on both sides together.

## 8. Running it

Same as the original (see the folder `README.md`). From the repository root,
with the environment set up:

```bash
cd automatic_door/automatic_labeling
../.venv/bin/python data_collection_improved.py
```

Output lands in `automatic_labeling/live_output/`, ready for
`build_masterdata.py` in the cleaning pipeline.
