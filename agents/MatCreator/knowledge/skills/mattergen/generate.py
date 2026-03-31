#!/usr/bin/env python3
"""Run MatterGen generation with a timestamped output directory.

This script is intended to be called from the MatCreator workspace, where the
current working directory acts as the output root. It creates a timestamped
directory, launches MatterGen from the configured virtual environment, and
writes the combined stdout/stderr stream to ``generation.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENT_ROOT = _SCRIPT_DIR.parents[3]
load_dotenv(_AGENT_ROOT / ".env", override=True)
_OUTPUT_ROOT = Path("/tmp/mattergen")


def _timestamped_output_dir(step_name: str) -> Path:
    now = datetime.now()
    dirname = (
        f"{now.strftime('%Y%m%d%H%M%S')}."
        f"{step_name}."
        f"{now.strftime('%f')}"
    )
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = _OUTPUT_ROOT / dirname
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate structures with MatterGen into a timestamped directory."
    )
    parser.add_argument(
        "--step-name",
        default="mattergen_generate",
        help="Label used in the timestamped output directory name.",
    )
    parser.add_argument(
        "--pretrained-name",
        required=True,
        help="Official MatterGen pretrained checkpoint name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size passed to mattergen-generate.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to generate.",
    )
    parser.add_argument(
        "--properties-to-condition-on",
        default=None,
        help="Raw MatterGen conditioning dictionary string.",
    )
    parser.add_argument(
        "--diffusion-guidance-factor",
        type=float,
        default=2,
        help="Optional diffusion guidance factor.",
    )
    return parser


def _resolve_mattergen_binary() -> Path:
    env_root = os.environ.get("MATTERGEN_ENV", "").strip()
    if not env_root:
        raise RuntimeError("MATTERGEN_ENV is not set in agents/MatCreator/.env.")

    binary = Path(env_root).expanduser().resolve() / "bin" / "mattergen-generate"
    if not binary.exists():
        raise RuntimeError(f"MatterGen executable not found: {binary}")
    return binary


def _build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [str(_resolve_mattergen_binary()), str(output_dir)]
    cmd.extend(["--pretrained-name", args.pretrained_name])

    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--num_batches", str(args.num_batches)])

    if args.properties_to_condition_on:
        cmd.extend(["--properties_to_condition_on", args.properties_to_condition_on])
    if args.diffusion_guidance_factor is not None:
        cmd.extend(
            ["--diffusion_guidance_factor", str(args.diffusion_guidance_factor)]
        )
    return cmd


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        output_dir = _timestamped_output_dir(args.step_name)
        log_path = output_dir / "generation.log"
        cmd = _build_command(args, output_dir)

        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path.cwd(),
            )

        response: dict[str, Any] = {
            "status": "success" if result.returncode == 0 else "error",
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            "returncode": result.returncode,
        }

        generated_extxyz = output_dir / "generated_crystals.extxyz"
        generated_cif_zip = output_dir / "generated_crystals_cif.zip"
        generated_traj_zip = output_dir / "generated_trajectories.zip"
        if generated_extxyz.exists():
            response["generated_extxyz"] = str(generated_extxyz)
        if generated_cif_zip.exists():
            response["generated_cif_zip"] = str(generated_cif_zip)
        if generated_traj_zip.exists():
            response["generated_trajectories_zip"] = str(generated_traj_zip)

        if result.returncode != 0:
            try:
                tail = log_path.read_text(encoding="utf-8")[-2000:]
            except Exception:
                tail = ""
            response["message"] = "MatterGen generation failed. See generation.log."
            response["log_tail"] = tail
            print(json.dumps(response, ensure_ascii=True))
            return result.returncode or 1

        response["message"] = "MatterGen generation completed successfully."
        print(json.dumps(response, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "output_dir": None,
                    "log_file": None,
                    "returncode": 1,
                },
                ensure_ascii=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
