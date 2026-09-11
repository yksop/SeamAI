# rq2_perception_clips/ — RQ2 perception/pipeline clips (steps 5-7)

The perception and pipeline benchmarks (steps 5-7) run YOLO on real frames, so
they need a folder of `.mp4` clips. This is that folder. It ships with just this
README and `select_clips.py`; the clips themselves are video of people and are
not committed.

## What goes here

A stratified sample of dataset clips — the **same recordings RQ1 used** (from
`MasterData/`), the `.mp4` video this time rather than the extracted
trajectories. Fill it with the script in this folder (run from the
`2_RQ2_RQ3_edge_deployment/` folder):

```bash
python benchmark_videos/rq2_perception_clips/select_clips.py          # ~30 clips
python benchmark_videos/rq2_perception_clips/select_clips.py --n 50   # more clips
```

The benchmarks then read from here:

```bash
python step5_benchmark_perception.py  --input-dir benchmark_videos/rq2_perception_clips/
python step6_benchmark_pipeline.py    --input-dir benchmark_videos/rq2_perception_clips/
python step7_benchmark_model_sizes.py --input-dir benchmark_videos/rq2_perception_clips/
```

`run_all.py` picks this folder up automatically. Around 20-30 clips spanning both
classes and several sessions is enough; each benchmark discards 50 warmup frames
and then times 500.

## Not the RQ3 clip

Step 9 (RQ3) uses a different input — one freshly recorded clip in
`../rq3_system_clip/`. Keep the two apart so steps 5-7 don't benchmark on the RQ3
clip.
