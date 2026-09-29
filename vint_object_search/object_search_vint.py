import argparse
import base64
import io
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
import torch
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from safetensors.torch import load_file


GOAL_MAX_FORWARD_IN = 150
GOAL_MAX_TURN_DEG = 45
GOAL_MIN_CONFIDENCE = 0.35
GOAL_STOP_DISTANCE_IN = 15
# Also treat the target as reached when its bounding box is this tall (fraction of frame height),
GOAL_STOP_BOX_HEIGHT = 0.55
# or when it drops out of view right after being seen within these limits (it slid below/beside
# the low camera as the rover closed in).
NEAR_LOST_DISTANCE_IN = 60
NEAR_LOST_BOX_HEIGHT = 0.3
TURN_AROUND_DEG = 180
GOAL_PREVIEW_WIDTH = 42
GOAL_IMAGE_INSTRUCTIONS = (
    "Create a plausible next camera view from this exact input photograph. "
    "Preserve the same place, objects, layout, lighting, camera height, and lens. "
    "Only apply the requested forward movement and turn. Do not add, remove, "
    "move, duplicate, or alter objects; "
    "Keep the result photorealistic and first-person. "
    "If the requested view cannot be inferred from the image, return no image."
)
INSTRUCTION_PROMPT = f"""You are guiding a wheeled indoor rover that is searching for the named target object.
You see one first-person frame from the rover's low front camera.

Return only JSON with keys reasoning, target_visible, target_box, target_distance_in, forward_in,
turn_deg, turn_around, confidence, arrived.

reasoning: one or two short sentences. Describe the space (hallway, open room, doorway, dead end,
corner, ...), where the target is likely to be, and why you chose this move.

If the target IS visible:
- target_box is its bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000.
- target_distance_in is the straight-line distance in inches from the camera to the target. Judge it
  from the object's apparent size and how much floor is visible between the camera and the object.
- Turn toward it and choose a small useful step, never driving past or into it.
- Set arrived=true if it is within about {GOAL_STOP_DISTANCE_IN} inches, even if it is off-center or
  partly cut off at the edge or bottom of the frame. If arrived is true, forward_in and turn_deg are 0.

If the target is NOT visible, explore to find it (target_box null, target_distance_in -1):
- Think about where this kind of object is usually found (e.g. trash cans near desks, kitchens,
  doorways, and walls; chairs at tables) and which visible direction most likely leads there.
- Prefer moves that reveal new space: continue down a long hallway, go toward or through an open
  doorway, head into an unexplored room, or turn to look around a corner.
- In a large open room, turn to scan unseen parts before driving far.
- If you are at a dead end, facing a wall, or boxed in with no useful way forward, set turn_around=true
  (forward_in and turn_deg 0) so the rover turns 180 degrees.
- Use the recent moves below to avoid repeating yourself: do not undo the previous move or revisit an
  area you just came from unless nothing else is left.
- Always stay in open floor space and avoid walls, furniture, people, and drop-offs. Do not return
  zero motion just because the target is out of view.

Limits: forward_in is 0 to {GOAL_MAX_FORWARD_IN} inches; turn_deg is {-GOAL_MAX_TURN_DEG} to
{GOAL_MAX_TURN_DEG} degrees (negative is left, positive is right); turn_around is true or false.
Set target_visible=true only when the named object is identifiable in this image.
confidence: when the target is visible, how sure you are it is the target; when exploring, how sure
you are the move is safe and useful. If unsure the target is there, use confidence below
{GOAL_MIN_CONFIDENCE} and do not claim arrived.

Recent moves (oldest first):
{{recent_moves}}

Target object: {{target}}"""


def _source_root() -> Path:
    return Path(__file__).resolve().parent.parent / "visualnav-transformer"


sys.path.insert(0, str(_source_root() / "train"))
from vint_train.models.vint.vint import ViNT


def _image_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def print_image(image: Image.Image, label: str, width: int = GOAL_PREVIEW_WIDTH):
    image = image.convert("RGB")
    rows = max(1, round(width * image.height / image.width * 0.5))
    pixels = image.resize((width, rows * 2)).load()
    print(label)
    for y in range(0, rows * 2, 2):
        line = []
        for x in range(width):
            top = pixels[x, y]
            bottom = pixels[x, y + 1]
            line.append(
                f"\x1b[38;2;{top[0]};{top[1]};{top[2]}m"
                f"\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀"
            )
        print("".join(line) + "\x1b[0m")


def _optional_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _box_height(box):
    try:
        ymin, _, ymax, _ = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (ymin, ymax)) or ymax <= ymin:
        return None
    return round(min(1000.0, ymax) / 1000 - max(0.0, ymin) / 1000, 3)


def generate_goal_instructions(client, model: str, frame: Image.Image, target: str,
                               recent_moves=None):
    prompt = INSTRUCTION_PROMPT.format(
        target=target,
        recent_moves="\n".join(f"- {move}" for move in recent_moves or []) or "- none yet",
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=[_image_part(frame), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception:
        return None
    try:
        result = json.loads(response.text or "")
        forward = float(result["forward_in"])
        turn = float(result["turn_deg"])
        confidence = float(result["confidence"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not all(math.isfinite(value) for value in (forward, turn, confidence)):
        return None
    instruction = {
        "reasoning": str(result.get("reasoning") or "")[:300],
        "target_visible": bool(result.get("target_visible", False)),
        "forward_in": min(GOAL_MAX_FORWARD_IN, max(0.0, forward)),
        "turn_deg": min(GOAL_MAX_TURN_DEG, max(-GOAL_MAX_TURN_DEG, turn)),
        "confidence": min(1.0, max(0.0, confidence)),
        "arrived": bool(result.get("arrived", False)),
        "turn_around": bool(result.get("turn_around", False)),
        "target_distance_in": _optional_float(result.get("target_distance_in")),
        "target_box_height": _box_height(result.get("target_box")),
    }
    # Never step past the stop distance, so the generated goal view can't be beyond the object.
    distance = instruction["target_distance_in"]
    if instruction["target_visible"] and distance is not None:
        instruction["forward_in"] = min(
            instruction["forward_in"], max(0.0, distance - GOAL_STOP_DISTANCE_IN)
        )
    return instruction


def generate_goal_image(client, model: str, frame: Image.Image, target: str, instruction: dict):
    if instruction["confidence"] < GOAL_MIN_CONFIDENCE:
        return None
    prompt = (
        f"{GOAL_IMAGE_INSTRUCTIONS}"
        f"Move forward {instruction['forward_in']:.0f} inches and turn "
        f"{instruction['turn_deg']:+.0f} degrees."
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=[_image_part(frame), prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
    except Exception:
        return None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                try:
                    image = Image.open(io.BytesIO(inline.data)).convert("RGB")
                    print_image(image, f"Generated goal image for {target}")
                    return image
                except (OSError, TypeError):
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(inline.data))).convert("RGB")
                        print_image(image, f"Generated goal image for {target}")
                        return image
                    except (OSError, TypeError, ValueError):
                        continue
    return None


def load_vint(config: dict, config_path: Path, device: torch.device):
    weights_path = (config_path.parent / config["vint"]["checkpoint_path"]).resolve()
    model = ViNT(
        context_size=5, len_traj_pred=5, learn_angle=True,
        obs_encoder="efficientnet-b0", obs_encoding_size=512, late_fusion=False,
        mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4,
    )
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    return model.to(device).eval()


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    image = ImageOps.fit(image.convert("RGB"), (85, 64), centering=(0.5, 0.5))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).to(device)


@torch.inference_mode()
def predict(model, observations, goal: Image.Image, device: torch.device):
    obs = torch.cat([_tensor(frame, device) for frame in observations]).unsqueeze(0)
    goal_tensor = _tensor(goal, device).unsqueeze(0)
    distance, actions = model(obs, goal_tensor)
    return float(distance[0].item()), actions[0].cpu().numpy()


def _control_from_waypoint(waypoint, config: dict, rate_hz: float, index: int):
    """ViNT's deployment pd_controller, returning SDK (linear, angular) and the (v, w) behind them.

    ViNT's controller divides by one tick, assuming a fresh frame every tick. The SDK's
    camera and commands lag by around a second, so that gain keeps the rover turning at
    full rate long after it faces the waypoint. This instead reaches the waypoint over the
    time ViNT predicts it takes: index + 1 ticks.
    """
    sdk = config["sdk"]
    max_v, max_w = sdk["max_v_mps"], sdk["max_w_radps"]
    dt = 1.0 / rate_hz
    horizon = (index + 1) * dt
    dx, dy, heading_x, heading_y = map(float, waypoint)
    # ViNT waypoints are normalized so one unit is the distance covered in one tick at max_v.
    dx, dy = dx * max_v * dt, dy * max_v * dt
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        v, w = 0.0, math.atan2(heading_y, heading_x) / horizon
    elif abs(dx) < 1e-8:
        v, w = 0.0, math.copysign(math.pi / (2 * horizon), dy)
    else:
        v, w = dx / horizon, math.atan(dy / dx) / horizon
    v = min(max_v, max(0.0, v))
    w = min(max_w, max(-max_w, w))
    linear = v / max_v * sdk["max_linear"]
    # The rover barely rotates below min_angular, so real turn requests start there and
    # scale up to max_angular; w below turn_deadband_radps is noise and drives straight.
    min_angular = sdk.get("min_angular", 0.0)
    if abs(w) < sdk.get("turn_deadband_radps", 0.05):
        angular = 0.0
    else:
        angular = math.copysign(
            min_angular + (sdk["max_angular"] - min_angular) * abs(w) / max_w, w)
    return max(-1.0, min(1.0, linear)), max(-1.0, min(1.0, angular)), v, w


class CameraStream:
    def __init__(self, base_url: str, fps: int = 10):
        self.response = requests.get(
            f"{base_url.rstrip('/')}/feed",
            params={"view": "front", "fps": fps},
            stream=True,
            timeout=(5, None),
        )
        self.response.raise_for_status()
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        buffer = bytearray()
        try:
            for chunk in self.response.iter_content(chunk_size=8192):
                if self.stop_event.is_set():
                    break
                if not chunk:
                    continue
                buffer.extend(chunk)
                while True:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end < 0:
                        break
                    headers = buffer[:header_end].decode("ascii", errors="ignore")
                    length = next(
                        (int(line.split(":", 1)[1]) for line in headers.split("\r\n")
                         if line.lower().startswith("content-length:")),
                        None,
                    )
                    if length is None:
                        raise ValueError("Camera stream part omitted Content-Length")
                    image_start = header_end + 4
                    image_end = image_start + length
                    if len(buffer) < image_end + 2:
                        break
                    jpeg = bytes(buffer[image_start:image_end])
                    del buffer[:image_end + 2]
                    with Image.open(io.BytesIO(jpeg)) as image:
                        frame = image.convert("RGB")
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                    self.frames.put_nowait(frame)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc

    def read(self, timeout: float = 5.0) -> Image.Image:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty as exc:
            if self.error:
                raise RuntimeError(f"Front camera stream failed: {self.error}") from self.error
            raise TimeoutError("Timed out waiting for a front camera frame") from exc

    def close(self):
        self.stop_event.set()
        self.response.close()
        self.thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def get_back_frame(base_url: str):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/v2/rear", timeout=5)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Image.open(io.BytesIO(base64.b64decode(response.json()["rear_frame"]))).convert("RGB")
    except requests.RequestException:
        return None


def send_control(base_url: str, linear: float, angular: float):
    requests.post(
        f"{base_url.rstrip('/')}/control",
        json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
        timeout=3,
    ).raise_for_status()


class ControlStreamer:
    def __init__(self, base_url: str, rate_hz: float):
        self.base_url = base_url
        self.interval = 1.0 / rate_hz
        self.command = (0.0, 0.0)
        self.lock = threading.Lock()
        self.changed = threading.Event()
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set(self, linear: float, angular: float):
        with self.lock:
            self.command = (linear, angular)
        self.changed.set()

    def _run(self):
        while not self.stopped.is_set():
            self.changed.wait(self.interval)
            self.changed.clear()
            if self.stopped.is_set():
                break
            with self.lock:
                linear, angular = self.command
            try:
                send_control(self.base_url, linear, angular)
            except Exception as exc:
                self.error = exc
                self.stopped.set()

    def close(self):
        self.stopped.set()
        self.changed.set()
        self.thread.join(timeout=1)
        send_control(self.base_url, 0.0, 0.0)


def rotate_in_place(base_url: str, config: dict, dry_run: bool, left_deg: float):
    """Rotate by `left_deg` degrees; positive is left (the SDK's positive angular)."""
    sdk = config["sdk"]
    seconds = math.radians(abs(left_deg)) / sdk["max_w_radps"]
    angular = math.copysign(sdk["max_angular"], left_deg)
    if dry_run:
        return
    deadline = time.monotonic() + seconds
    interval = 1.0 / sdk["control_hz"]
    try:
        while time.monotonic() < deadline:
            send_control(base_url, 0.0, angular)
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    finally:
        send_control(base_url, 0.0, 0.0)


def turn_around(base_url: str, config: dict, dry_run: bool):
    print(f"Turning rover {TURN_AROUND_DEG} degrees before using the rear-camera goal")
    rotate_in_place(base_url, config, dry_run, TURN_AROUND_DEG)


class ArrivalTracker:
    """Decides from successive Gemini checks whether the rover has reached the target."""

    def __init__(self):
        self.was_near = False

    def update(self, instruction):
        """Returns why the target counts as reached, or None."""
        if instruction is None:
            return None
        distance = instruction["target_distance_in"]
        height = instruction["target_box_height"]
        if instruction["target_visible"]:
            if instruction["confidence"] < GOAL_MIN_CONFIDENCE:
                return None
            self.was_near = (
                (distance is not None and distance <= NEAR_LOST_DISTANCE_IN)
                or (height is not None and height >= NEAR_LOST_BOX_HEIGHT)
            )
            if instruction["arrived"]:
                return f"Gemini says arrived (~{distance} in, box height {height})"
            if distance is not None and distance <= GOAL_STOP_DISTANCE_IN:
                return f"estimated {distance:.0f} in away (<= {GOAL_STOP_DISTANCE_IN})"
            if height is not None and height >= GOAL_STOP_BOX_HEIGHT:
                return f"fills {height:.0%} of frame height (>= {GOAL_STOP_BOX_HEIGHT:.0%})"
            return None
        if self.was_near:
            return "lost from view right after being close"
        return None


class ArrivalChecker:
    """Runs Gemini arrival checks on the newest frame in the background so the drive loop
    never waits on Gemini."""

    def __init__(self, client, model: str, target: str):
        self.client, self.model, self.target = client, model, target
        self.frame = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.results = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame: Image.Image):
        with self.lock:
            self.frame = frame
        self.ready.set()

    def drain(self):
        results = []
        while True:
            try:
                results.append(self.results.get_nowait())
            except queue.Empty:
                return results

    def _run(self):
        while not self.stopped.is_set():
            self.ready.wait()
            self.ready.clear()
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None or self.stopped.is_set():
                continue
            self.results.put(generate_goal_instructions(self.client, self.model, frame, self.target))

    def close(self):
        self.stopped.set()
        self.ready.set()


def _drive_goal(model, history, goal, config, device, base_url, camera, dry_run, client, target):
    """Drive toward `goal` with ViNT, streaming commands continuously like ViNT's deployment,
    until the rover reaches the goal image or Gemini sees the target object.
    Returns (object_found, message)."""
    vint_config = config["vint"]
    rate_hz = vint_config.get("rate_hz", 4.0)
    threshold = vint_config.get("goal_reached_threshold", 3.0)
    overshoot_margin = vint_config.get("overshoot_margin", 1.0)
    overshoot_patience = vint_config.get("overshoot_patience", 8)
    index = vint_config.get("waypoint_index", 2)
    streamer = None if dry_run else ControlStreamer(base_url, config["sdk"]["control_hz"])
    checker = ArrivalChecker(client, config["vlm"]["reasoning_model"], target)
    tracker = ArrivalTracker()
    step = 0
    best_distance, best_step, rising = float("inf"), 0, 0
    next_tick = time.monotonic()
    try:
        while True:
            step += 1
            frame = camera.read()
            history.append(frame)
            checker.submit(frame)
            for check in checker.drain():
                print(f"  Gemini check: {check if check is not None else 'no valid instruction'}")
                reached = tracker.update(check)
                if reached:
                    return True, f"{reached} at ViNT step {step}"
            distance, actions = predict(model, list(history), goal, device)
            if distance <= threshold:
                return False, (f"reached Gemini-generated goal image at ViNT step {step} "
                        f"(distance={distance:.2f} <= {threshold})")
            # ViNT's distance to a generated image often bottoms out above the threshold;
            # if it stays above its minimum, the rover has passed the goal view.
            if distance < best_distance:
                best_distance, best_step, rising = distance, step, 0
            elif distance > best_distance + overshoot_margin:
                rising += 1
                if rising >= overshoot_patience:
                    return False, (f"passed Gemini-generated goal image (closest d={best_distance:.2f} "
                            f"at step {best_step}, now d={distance:.2f} at step {step})")
            index = min(index, len(actions) - 1)
            waypoint = actions[index]
            linear, angular, v, w = _control_from_waypoint(waypoint, config, rate_hz, index)
            print(f"ViNT step {step}: d={distance:.2f} "
                  f"wp=[{', '.join(f'{value:+.2f}' for value in waypoint)}] "
                  f"v={v:.2f}m/s w={w:+.2f}rad/s (+ is left) cmd=({linear:.3f}, {angular:+.3f})")
            if streamer:
                if streamer.error:
                    raise RuntimeError(f"Control stream failed: {streamer.error}")
                streamer.set(linear, angular)
            next_tick += 1.0 / rate_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))
            next_tick = max(next_tick, time.monotonic())
    finally:
        checker.close()
        if streamer:
            streamer.close()


def _stop_rover(message: str, base_url: str, config: dict, dry_run: bool):
    print(f"\n{message}; stopping rover")
    if not dry_run:
        send_control(base_url, 0.0, 0.0)
        time.sleep(config["loop"].get("post_found_pause_s", 0.0))


def run(config_path: Path, dry_run: bool = False):
    load_dotenv(config_path.parent / ".env")
    with config_path.open() as file:
        config = yaml.safe_load(file)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    device = torch.device(config["vint"].get("device", "cpu"))
    model = load_vint(config, config_path, device)
    history = deque(maxlen=model.context_size + 1)
    base_url = config["sdk"]["base_url"]
    vlm = config["vlm"]
    period = 1.0 / config["loop"]["control_loop_hz"]

    camera = CameraStream(base_url, config["sdk"].get("camera_stream_fps", 10))
    try:
        objects = config["objects"]
        for number, target in enumerate(objects, 1):
            print(f"\n=== Searching for object {number}/{len(objects)}: {target} ===")
            arrived_count = 0
            tracker = ArrivalTracker()
            recent_moves = deque(maxlen=8)
            rear_turn_done = False
            rear_goal = None
            cycle = 0
            history.clear()
            while arrived_count < config["loop"].get("found_confirm_count", 2):
                frame = camera.read()
                history.append(frame)
                if len(history) < history.maxlen:
                    print(f"{target}: buffering camera history {len(history)}/{history.maxlen}")
                    if not dry_run:
                        send_control(base_url, 0.0, 0.0)
                    time.sleep(period)
                    continue

                if rear_goal is not None:
                    found, reason = _drive_goal(model, history, rear_goal, config, device, base_url,
                                                camera, dry_run, client, target)
                    rear_goal = None
                    if found:
                        _stop_rover(f"{target}: OBJECT REACHED -- {reason}", base_url, config, dry_run)
                        break
                    _stop_rover(f"{target}: GOAL IMAGE REACHED -- {reason}; regenerating "
                                "instructions and goal image", base_url, config, dry_run)
                    continue

                front_instruction = generate_goal_instructions(
                    client, vlm["reasoning_model"], frame, target, recent_moves
                )
                back_frame = get_back_frame(base_url)
                back_instruction = (
                    generate_goal_instructions(client, vlm["reasoning_model"], back_frame, target)
                    if back_frame is not None else None
                )
                cycle += 1
                print(f"\n{target}: Gemini instruction cycle {cycle}")
                print(f"  front: {front_instruction if front_instruction is not None else 'no valid instruction'}")
                if back_frame is not None:
                    print(f"  rear:  {back_instruction if back_instruction is not None else 'no valid instruction'}")
                else:
                    print("  rear:  unavailable")
                if (
                    not rear_turn_done
                    and back_instruction is not None
                    and back_instruction["target_visible"]
                    and back_instruction["confidence"] >= GOAL_MIN_CONFIDENCE
                    and not (front_instruction and front_instruction["target_visible"])
                ):
                    rear_goal = generate_goal_image(
                        client, vlm["goal_image_model"], back_frame, target, back_instruction
                    )
                    if rear_goal is not None:
                        turn_around(base_url, config, dry_run)
                        history.clear()
                        rear_turn_done = True
                        recent_moves.append("turned around because the rear camera saw the target")
                        continue

                instruction = front_instruction
                reached = tracker.update(instruction)
                arrived_count = arrived_count + 1 if reached else 0
                if arrived_count >= config["loop"].get("found_confirm_count", 2):
                    _stop_rover(f"{target}: OBJECT REACHED -- {reached} "
                                f"(confirmed {arrived_count}x)", base_url, config, dry_run)
                    break
                if reached:
                    print(f"{target}: possibly reached ({reached}); re-checking to confirm")
                    if not dry_run:
                        send_control(base_url, 0.0, 0.0)
                    time.sleep(period)
                    continue
                if instruction is None or instruction["confidence"] < GOAL_MIN_CONFIDENCE:
                    if not dry_run:
                        send_control(base_url, 0.0, 0.0)
                    time.sleep(period)
                    continue
                if instruction["turn_around"] and not instruction["target_visible"]:
                    print(f"{target}: Gemini chose to turn around ({instruction['reasoning']})")
                    turn_around(base_url, config, dry_run)
                    history.clear()
                    recent_moves.append(f"turned around 180 deg: {instruction['reasoning']}")
                    continue
                if instruction["forward_in"] < 1 and abs(instruction["turn_deg"]) < 1:
                    print(f"{target}: Gemini proposed no motion (visible={instruction['target_visible']}, "
                          f"arrived={instruction['arrived']}); requesting a fresh instruction")
                    if not dry_run:
                        send_control(base_url, 0.0, 0.0)
                    time.sleep(period)
                    continue

                goal = generate_goal_image(client, vlm["goal_image_model"], frame, target, instruction)
                if goal is None:
                    if not dry_run:
                        send_control(base_url, 0.0, 0.0)
                    time.sleep(period)
                    continue
                found, reason = _drive_goal(model, history, goal, config, device, base_url,
                                            camera, dry_run, client, target)
                recent_moves.append(
                    f"moved ~{instruction['forward_in']:.0f} in, turned {instruction['turn_deg']:+.0f} deg: "
                    f"{instruction['reasoning']}"
                )
                if found:
                    _stop_rover(f"{target}: OBJECT REACHED -- {reason}", base_url, config, dry_run)
                    break
                _stop_rover(f"{target}: GOAL IMAGE REACHED -- {reason}; regenerating "
                            "instructions and goal image", base_url, config, dry_run)
        print(f"\nAll {len(objects)} objects found; stopping")
    finally:
        camera.close()
        if not dry_run:
            try:
                send_control(base_url, 0.0, 0.0)
            except requests.RequestException:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.dry_run)


