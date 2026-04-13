"""
run_training.py
===============
Self-contained launcher for train_fsdp_ipi_2.py.

Steps
-----
1. Install / upgrade all required Python packages.
2. Login to Weights & Biases.
3. Set W&B environment variables.
4. Login to HuggingFace Hub.
5. Launch training via `accelerate launch`.

Usage (Kaggle / Colab notebook cell, or terminal)
-------------------------------------------------
    python run_training.py \
        --hf_token    YOUR_HF_TOKEN \
        --hf_username YOUR_HF_USERNAME \
        --wandb_key   YOUR_WANDB_KEY \
        --model_id    meta-llama/Llama-2-7b-chat-hf \
        --output_dir  /kaggle/working/checkpoints/llama2-7b-ipi-qlora

Optional overrides
------------------
    --dataset_path   datasets/ipi                       (default)
    --config         fine-tuning/fsdp_lora_dafaults.yaml (default)
    --training_mode  qlora                              (default)
    --wandb_project  ISEC-exp-1                         (default)
    --hf_repo_id     username/my-custom-repo-name       (auto-derived from output_dir if omitted)
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training launcher for train_fsdp_ipi_2.py")

    # Required credentials
    p.add_argument("--hf_token",    required=True,  help="HuggingFace API token")
    p.add_argument("--hf_username", required=True,  help="HuggingFace username (for Hub push)")
    p.add_argument("--wandb_key",   required=True,  help="Weights & Biases API key")

    # Model / data
    p.add_argument("--model_id",    required=True,  help="HuggingFace model id, e.g. meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--output_dir",  required=True,  help="Directory to save checkpoints")

    # Optional overrides
    p.add_argument("--dataset_path",    default="datasets/ipi",                         help="Path to IPI dataset directory")
    p.add_argument("--train_config",    default="fine-tuning/fsdp_lora_dafaults.yaml",  help="Training config yaml passed to train_fsdp_ipi_2.py (SFTConfig)")
    p.add_argument("--accelerate_config",default=None,                                  help="Accelerate launcher config yaml (optional). Leave empty for single-GPU defaults.")
    p.add_argument("--training_mode",   default="qlora",                                help="lora | qlora | fft")
    p.add_argument("--wandb_project",   default="ISEC-exp-1",                           help="W&B project name")
    p.add_argument("--hf_repo_id",      default=None,                                   help="Full Hub repo id (optional; derived from output_dir if omitted)")
    p.add_argument("--test",            action="store_true",                             help="Smoke-test mode: run only 5 training steps")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and raise on non-zero exit."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Install / upgrade dependencies
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Installing / upgrading dependencies...")
    print("=" * 60)

    packages = [
        "torch", "torchvision", "torchaudio",
        "datasets", "transformers", "trl", "peft",
        "accelerate", "bitsandbytes",
        "huggingface_hub",
        "numpy", "pandas", "scipy", "tqdm",
        "wandb",
    ]

    run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"] + packages)
    print("Dependencies installed successfully.\n")

    # ------------------------------------------------------------------
    # 2. W&B login + environment variables
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Configuring Weights & Biases...")
    print("=" * 60)

    import wandb  # noqa: PLC0415 – imported after pip install

    wandb.login(key=args.wandb_key)
    os.environ["WANDB_PROJECT"]  = args.wandb_project
    os.environ["WANDB_DISABLED"] = "false"
    print(f"W&B project: {args.wandb_project}\n")

    # ------------------------------------------------------------------
    # 3. HuggingFace login
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Logging in to HuggingFace Hub...")
    print("=" * 60)

    from huggingface_hub import login  # noqa: PLC0415

    login(token=args.hf_token)
    print("HuggingFace login successful.\n")

    # ------------------------------------------------------------------
    # 4. Build the accelerate launch command
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Launching training...")
    print("=" * 60)

    import shutil  # noqa: PLC0415
    accelerate_bin = shutil.which("accelerate") or os.path.join(
        os.path.dirname(sys.executable), "accelerate"
    )

    cmd = [accelerate_bin, "launch"]

    # Only pass --config_file if the user provided an accelerate launcher config
    if args.accelerate_config:
        cmd += ["--config_file", args.accelerate_config]

    cmd += [
        "fine-tuning/train_fsdp_ipi_2.py",
        "--config",          args.train_config,   # training / SFTConfig yaml
        "--training_mode",   args.training_mode,
        "--dataset_path",    args.dataset_path,
        "--model_id",        args.model_id,
        "--output_dir",      args.output_dir,
        "--hf_token",        args.hf_token,
        "--hf_username",     args.hf_username,
    ]

    if args.hf_repo_id:
        cmd += ["--hf_repo_id", args.hf_repo_id]

    if args.test:
        cmd += ["--max_steps", "5"]
        print("[TEST MODE] Training limited to 5 steps.")

    run(cmd)

    print("\n" + "=" * 60)
    print("Training complete!")

    # Derive repo id so we can print the Hub URL
    repo_id = args.hf_repo_id or (
        f"{args.hf_username}/{os.path.basename(args.output_dir.rstrip('/'))}"
    )
    print(f"Model pushed to : https://huggingface.co/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
