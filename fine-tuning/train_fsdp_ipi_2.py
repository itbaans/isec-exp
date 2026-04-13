"""
train_fsdp_ipi.py

Fine-tuning script for IPI (Indirect Prompt Injection) detection.
Supports: gemma-1.1, gemma-2, Phi-3, Llama-3

Loads data produced by prepare_dataset_ipi.py (train_dataset_ipi.json / test_dataset_ipi.json).

The text field in each record is already formatted with the model's chat template,
so SFTTrainer uses the 'text' field directly (dataset_text_field='text').

Usage (example with accelerate + FSDP)
---------------------------------------
    accelerate launch --config_file fsdp_config.yaml train_fsdp_ipi.py \
        --dataset_path ../../datasets/ipi \
        --model_id     google/gemma-1.1-7b-it \
        --output_dir   ./checkpoints/gemma1.1-ipi-lora \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-4

    # Phi-3
    accelerate launch --config_file fsdp_config.yaml train_fsdp_ipi.py \
        --dataset_path ../../datasets/ipi \
        --model_id     microsoft/Phi-3-mini-4k-instruct \
        --output_dir   ./checkpoints/phi3-ipi-lora

    # Llama-3
    accelerate launch --config_file fsdp_config.yaml train_fsdp_ipi.py \
        --dataset_path ../../datasets/ipi \
        --model_id     meta-llama/Meta-Llama-3-8B-Instruct \
        --output_dir   ./checkpoints/llama3-ipi-lora
"""

import os
os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from dataclasses import dataclass, field
import random

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import TrlParser, SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Per-model default LoRA target modules
# (can be overridden via --peft_target_modules)
# ---------------------------------------------------------------------------

DEFAULT_LORA_TARGETS = {
    # Gemma shares the same projection names
    "gemma":  ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
    # Phi-3 uses different names; "all-linear" also works but explicit is safer
    "phi":    ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    # Llama-3 same as Gemma naming convention
    "llama":  ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
}

# Phi-3 requires trust_remote_code=True
TRUST_REMOTE_CODE_PREFIXES = ("microsoft/Phi-3",)


def _needs_trust_remote_code(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in TRUST_REMOTE_CODE_PREFIXES)


def _default_lora_targets(model_id: str) -> list[str]:
    """Infer sensible LoRA target modules from the model_id string."""
    mid = model_id.lower()
    if "phi" in mid:
        return DEFAULT_LORA_TARGETS["phi"]
    if "llama" in mid:
        return DEFAULT_LORA_TARGETS["llama"]
    # Default: Gemma (and anything else)
    return DEFAULT_LORA_TARGETS["gemma"]


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    """
    Arguments specific to this IPI fine-tuning script.

    Attributes
    ----------
    dataset_path        : Directory containing train_dataset_ipi.json / test_dataset_ipi.json.
    model_id            : HuggingFace model ID (Gemma, Phi-3, or Llama-3 variant).
    hf_token            : HuggingFace API token for private/gated model access. Optional.
    hf_username         : HuggingFace username used when pushing the model to the Hub.
    hf_repo_id          : Full repo id (username/repo_name) to push the model to.
                          Defaults to {hf_username}/{output_dir_basename} when not set.
    max_seq_length      : Maximum token sequence length fed to SFTTrainer. Default: 2048.
    training_mode       : One of 'lora', 'qlora', 'fft'. Default: 'lora'.
    attention_impl      : Attention implementation: 'sdpa' or 'flash_attention_2'. Default: 'sdpa'.
    lora_r              : LoRA rank. Default: 16.
    lora_alpha          : LoRA alpha scaling. Default: 32.
    lora_dropout        : LoRA dropout probability. Default: 0.05.
    peft_target_modules : Modules to apply LoRA to. Inferred from model_id if not set.
    """

    dataset_path: str = field(
        default=None,
        metadata={"help": "Directory with train_dataset_ipi.json and test_dataset_ipi.json"},
    )
    model_id: str = field(
        default="google/gemma-1.1-7b-it",
        metadata={"help": "HuggingFace model ID (gemma, Phi-3, or Llama-3 variant)"},
    )
    hf_token: str = field(
        default=None,
        metadata={"help": "HuggingFace API token for gated/private model access"},
    )
    hf_username: str = field(
        default=None,
        metadata={"help": "HuggingFace username (required when pushing model to Hub)"},
    )
    hf_repo_id: str = field(
        default=None,
        metadata={
            "help": (
                "Full HuggingFace repo id to push to (e.g. username/my-model). "
                "Defaults to {hf_username}/{output_dir_basename} when not provided."
            )
        },
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for SFTTrainer"},
    )
    training_mode: str = field(
        default="lora",
        metadata={"help": "Training mode: lora | qlora | fft"},
    )
    attention_impl: str = field(
        default="sdpa",
        metadata={"help": "Attention implementation: sdpa | flash_attention_2"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    peft_target_modules: list[str] = field(
        default=None,
        metadata={
            "help": (
                "LoRA target modules. When None (default), sensible defaults are "
                "chosen automatically based on model_id."
            )
        },
    )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def training_function(
    script_args: ScriptArguments,
    training_args: SFTConfig,
) -> None:
    """
    Load the IPI dataset, initialise the model (with optional LoRA / QLoRA),
    and run SFTTrainer.

    Steps
    -----
    1. Load train_dataset_ipi.json and test_dataset_ipi.json.
    2. Initialise tokenizer.
    3. Log a few random training samples.
    4. Initialise model (with optional 4-bit quantization for qlora).
    5. Configure LoRA / QLoRA PEFT if requested.
    6. Train with SFTTrainer and save the model.
    """

    # ------------------------------------------------------------------
    # 0. HuggingFace login (if token provided)
    # ------------------------------------------------------------------
    if script_args.hf_token:
        print("Logging in to HuggingFace Hub...")
        login(token=script_args.hf_token)
        print("HuggingFace login successful.")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset_ipi.json"),
        split="train",
    )

    print(f"Train samples : {len(train_dataset)}")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_id,
        use_fast=True,
        trust_remote_code=_needs_trust_remote_code(script_args.model_id),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. Log random samples
    # ------------------------------------------------------------------
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), min(2, len(train_dataset))):
            sample_text = train_dataset[index][training_args.dataset_text_field]
            print(f"\n--- Training sample {index} ---")
            print(sample_text[:600])
            print("...")

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    torch_dtype         = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    if script_args.training_mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation=script_args.attention_impl,
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=_needs_trust_remote_code(script_args.model_id),
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # 5. PEFT / LoRA
    # ------------------------------------------------------------------
    if script_args.training_mode in ("lora", "qlora"):
        # Resolve target modules: explicit arg > auto-detected defaults
        target_modules = (
            script_args.peft_target_modules
            if script_args.peft_target_modules is not None
            else _default_lora_targets(script_args.model_id)
        )
        print(f"LoRA target modules: {target_modules}")

        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # ------------------------------------------------------------------
    # 6. SFTTrainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if (
        trainer.accelerator.is_main_process
        and hasattr(trainer.model, "print_trainable_parameters")
    ):
        trainer.model.print_trainable_parameters()

    # Resume from checkpoint if specified
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save (handles FSDP full-state-dict consolidation automatically)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    # ------------------------------------------------------------------
    # 7. Push LoRA adapter to HuggingFace Hub (main process only)
    # ------------------------------------------------------------------
    if trainer.accelerator.is_main_process and script_args.hf_username:
        repo_id = script_args.hf_repo_id or (
            f"{script_args.hf_username}/{os.path.basename(training_args.output_dir.rstrip('/'))}"
        )
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        trainer.model.push_to_hub(repo_id, private=True)
        tokenizer.push_to_hub(repo_id, private=True)
        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Force bf16 detection on Ampere+ GPUs (compute capability >= 8.0).
    # RTX 3090 (GA102, sm_86) has bf16 ALUs but PyTorch's
    # torch.cuda.is_bf16_supported() can return False in some environments,
    # causing transformers to reject bf16=True before training starts.
    # Patch it before TrlParser runs its validation.
    # ------------------------------------------------------------------
    import torch as _torch
    if _torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 8:
        _torch.cuda.is_bf16_supported = lambda: True

    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.dataset_text_field is None:
        training_args.dataset_text_field = "text"

    training_args.max_seq_length = script_args.max_seq_length
    training_args.packing = False
    training_args.dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token": False,
    }

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    set_seed(training_args.seed)

    training_function(script_args, training_args)