# VLM-guided ViNT object search

For each object in `config.yaml`, the loop reads the current front-camera
continuous front-camera stream (default 10 FPS) and, when available, the
rear-camera frame. Gemini receives separate,
identical instruction calls for each view. If only the rear view sees the
target, the loop generates its goal image from that frame, turns the rover
180 degrees, clears the old camera history, and feeds ViNT fresh front-camera
observations with that rear-derived goal. For each goal image, ViNT repeatedly
infers on fresh frames until its distance estimate is within
`vint.goal_reached_threshold`; Gemini then generates the next instruction and
goal. The rover stops after consecutive close-object confirmations. There is
no per-goal step limit, so use an independent physical safety stop.

Motion proposals are clamped to `0..150` inches forward and `-45..45`
degrees of turn. Low-confidence or malformed responses cause a stop. Goal
image instructions require preserving the original scene and prohibit adding
or removing objects; generated views can still be inaccurate, so begin with
`--dry-run`.

## Setup

```bash
# from vint_object_search/
pip install -r requirements.txt

export GEMINI_API_KEY=...          # Gemini reasoning and goal-image generation

# in another terminal: start the Earth Rovers SDK
cd ../earth-rovers-sdk && hypercorn main:app --reload
```

Edit [`config.yaml`](config.yaml) and set `objects` to your target list. The
ViNT uses the official checkpoint in the sibling `visualnav-transformer`
checkout. Calibrate `sdk.max_v_mps`, `sdk.max_w_radps`, `sdk.max_linear`, and
`sdk.max_angular` against the rover's real speed before enabling control.

## Run

```bash
python object_search_vint.py --config config.yaml

# or, to see the VLM/ViNT plan without actually driving the rover:
python object_search_vint.py --config config.yaml --dry-run
```

## Notes / limitations

- Generated goal views can alter scene details despite the restrictive prompt.
- The script loads the official checkpoint from the sibling
  `visualnav-transformer` checkout. Tune `sdk.max_*` for the physical rover.
- Requires a running Earth Rovers SDK server (see
  [`../earth-rovers-sdk/README.md`](../earth-rovers-sdk/README.md)) for the
  `/v2/front`, `/control` and `/speak` endpoints this script talks to.
