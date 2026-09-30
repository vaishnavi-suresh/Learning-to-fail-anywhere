"""object_search_vint.py, driven by the fine-tuned ViNT instead of the official release.

The model is the one notebook 04 fine-tunes (Section 9) and publishes (Section 10). It has
the official ViNT's architecture and I/O, so everything else -- Gemini goal images, the
controller, the rover loop -- is object_search_vint.py unchanged.

Weights come from the private HF repo (token from HF_TOKEN in a .env next to the config or
at the repo root, the environment, or `huggingface-cli login`),
or from a local file with --checkpoint.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import object_search_vint as base

FINETUNED_REPO = "vaishsuresh32/vint-sacson-finetuned"
VINT_KEYS = ("context_size", "len_traj_pred", "learn_angle", "obs_encoder", "obs_encoding_size",
             "late_fusion", "mha_num_attention_heads", "mha_num_attention_layers", "mha_ff_dim_factor")


def load_finetuned_vint(checkpoint: Path = None):
    def load(config: dict, config_path: Path, device: torch.device):
        if checkpoint is not None:
            weights_path = checkpoint
            vint_config = dict(context_size=5, len_traj_pred=5, learn_angle=True,
                               obs_encoder="efficientnet-b0", obs_encoding_size=512, late_fusion=False,
                               mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4,
                               image_size=[85, 64])
        else:
            # HF_TOKEN may live next to the config or in the repo-root .env.
            load_dotenv(config_path.parent / ".env")
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            repo = config["vint"].get("finetuned_repo", FINETUNED_REPO)
            token = (os.environ.get("HF_TOKEN") or "").strip().strip("'\"") or None
            weights_path = hf_hub_download(repo, "model.safetensors", token=token)
            with open(hf_hub_download(repo, "config.json", token=token)) as file:
                vint_config = json.load(file)
        # object_search_vint feeds ViNT 85x64 frames and a 5-waypoint controller.
        if vint_config.get("image_size", [85, 64]) != [85, 64] or vint_config["len_traj_pred"] != 5:
            raise ValueError(f"{weights_path} isn't in the official ViNT format: {vint_config}")
        model = base.ViNT(**{key: vint_config[key] for key in VINT_KEYS})
        model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
        print(f"Loaded fine-tuned ViNT from {checkpoint or repo}")
        return model.to(device).eval()

    return load


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        help="local fine-tuned .safetensors instead of downloading from HF")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    # GEMINI_API_KEY and HF_TOKEN live in the repo-root .env; run() only reads the one by the config.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    base.load_vint = load_finetuned_vint(arguments.checkpoint.resolve() if arguments.checkpoint else None)
    base.run(arguments.config.resolve(), arguments.dry_run)
