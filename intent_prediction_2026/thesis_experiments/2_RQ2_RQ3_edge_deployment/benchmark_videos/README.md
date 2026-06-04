# benchmark_videos/ — video inputs for the RQ2/RQ3 benchmarks

The benchmarks that run on real frames need video. There are two kinds, used by
different steps, so each lives in its own subfolder together with the script that
produces it:

```
benchmark_videos/
  rq2_perception_clips/   # steps 5-7 (RQ2): many short dataset clips
    select_clips.py       #   fills this folder from MasterData/
    README.md
  rq3_system_clip/        # step 9  (RQ3): one recorded reference clip
    record_clip.py        #   records rq3_clip.mp4 here
    verify_clip.py        #   checks the recorded clip
    README.md
```

- **rq2_perception_clips/** — a stratified sample of dataset clips (the same
  recordings RQ1 used), for the perception and pipeline benchmarks (steps 5-7).
  Fill it by running its `select_clips.py`.
- **rq3_system_clip/** — one freshly recorded clip of a person walking toward and
  away from the camera, for the end-to-end system-latency benchmark (step 9, RQ3).
  Record it with its `record_clip.py`.

The clips are video of people and are **not** committed to git — only these
READMEs and the helper scripts are. `run_all.py` auto-detects both folders, so
once they are populated you can simply run `python run_all.py`.
