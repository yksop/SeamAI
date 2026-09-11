# Demo — Open House

A quick live-camera demo built to draw attention at the RISE open house. It draws
a box around each person walking toward the door and shows a live **P(enter)** —
the model's guess of whether they'll come in or just pass by. It's a visual
showcase, not a rigorous part of the thesis.

Under the hood it runs the same kind of model as the experiments (a small
Core-3 + head GRU) on a YOLO-pose camera feed, on a laptop or the Jetson.

## Run

```bash
cd demo_open_house
python demo_live.py --model weights/gru_core3head.pt --norm weights/norm_stats.npz
```

The small trained model is included in `weights/`, so it runs out of the box. The
YOLO-pose model downloads automatically on first run. Run `python demo_live.py`
with no `--model` to just show the boxes and skeletons without predictions.
Press `q` to quit, `f` for fullscreen.

## Files

```
demo_open_house/
  demo_live.py         # the demo — run this
  model.py             # GRU definition + loading
  features.py          # the 7 features fed to the GRU
  train_demo_model.py  # (re)train the demo model from MasterData
  test_camera.py       # quick camera check
  weights/             # the small trained model (included)
```
