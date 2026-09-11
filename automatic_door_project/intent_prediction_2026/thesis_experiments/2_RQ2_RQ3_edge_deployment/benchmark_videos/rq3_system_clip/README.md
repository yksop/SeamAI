# rq3_system_clip/ — RQ3 reference clip (step 9)

Step 9 (RQ3, end-to-end system latency) needs a short recorded video so it can
time the full chain *including* camera capture and video decode — the parts the
steps 5-7 benchmarks skip. This folder holds that one clip plus the tools to make
and check it. It ships with just this README and the two scripts; the clip itself
is not committed.

## What goes here

A single reference clip, `rq3_clip.mp4` — a person walking toward and away from
the camera at a few distances and angles, recorded on the Jetson with the USB
camera mounted overhead (like a door sensor). The content is not labeled; step 9
only measures timing, so all that matters is that a person is in frame and YOLO
keeps detecting them. The thesis used a ~90 s clip (about 2,700 frames).

Record and check it with the scripts in this folder (run from the
`2_RQ2_RQ3_edge_deployment/` folder):

```bash
python benchmark_videos/rq3_system_clip/record_clip.py --preview      # set up the camera angle (optional)
python benchmark_videos/rq3_system_clip/record_clip.py --duration 90  # record rq3_clip.mp4 here
python benchmark_videos/rq3_system_clip/verify_clip.py                # check the detection rate first
```

The RQ3 benchmark then reads it:

```bash
python step9_benchmark_live_camera.py --video benchmark_videos/rq3_system_clip/rq3_clip.mp4
# or just let run_all find it:
python run_all.py
```

## Not the RQ2 clips

Steps 5-7 use many short dataset clips in `../rq2_perception_clips/`. This folder
holds one fresh recording for the live latency measurement (step 9). Keep them
separate.
