"""
train_fsdp_ipi.py

Fine-tuning script for IPI (Indirect Prompt Injection) detection on Gemma models.
Loads data produced by prepare_dataset_ipi.py (train_dataset_ipi.json / test_dataset_ipi.json).

The text field in each record is already formatted with the Gemma chat template:
    <start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n
    <start_of_turn>model\n{assistant}<end_of_turn>

SFTTrainer uses the 'text' field directly (dataset_text_field='text').

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
"""

import os
os.environ["BNB_CUDA_VERSION"] = "121"  # Forces bitsandbytes to use CUDA 12.1 binaries
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from dataclasses import dataclass, field
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import TrlParser, SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    """
    Arguments specific to this IPI fine-tuning script.

    Attributes
    ----------
    dataset_path        : Directory containing train_dataset_ipi.json and test_dataset_ipi.json.
    dataset_text_field  : Name of the field that holds the formatted chat text. Default: 'text'.
    model_id            : HuggingFace model ID (must be a Gemma variant).
    max_seq_length      : Maximum token sequence length fed to SFTTrainer. Default: 2048.
    training_mode       : One of 'lora', 'qlora', 'fft'. Default: 'lora'.
    attention_impl      : Attention implementation: 'sdpa' or 'flash_attention_2'. Default: 'sdpa'.
    lora_r              : LoRA rank. Default: 16.
    lora_alpha          : LoRA alpha scaling. Default: 32.
    lora_dropout        : LoRA dropout probability. Default: 0.05.
    peft_target_modules : Which modules to apply LoRA to. Default: 'all-linear'.
    """

    dataset_path: str = field(
        default=None,
        metadata={"help": "Directory with train_dataset_ipi.json and test_dataset_ipi.json"},
    )
    model_id: str = field(
        default="google/gemma-1.1-7b-it",
        metadata={"help": "HuggingFace model ID, e.g. google/gemma-1.1-7b-it"},
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
    peft_target_modules: str = field(
        default="all-linear",
        metadata={"help": "Which modules to target with LoRA"},
    )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def training_function(
    script_args: ScriptArguments,
    training_args: SFTConfig,
) -> None:
    """
    Load the IPI dataset, initialise the Gemma model (with optional LoRA / QLoRA),
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
    # 1. Dataset
    # ------------------------------------------------------------------
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset_ipi.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset_ipi.json"),
        split="train",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test  samples: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
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
    torch_dtype          = torch.bfloat16
    quant_storage_dtype  = torch.bfloat16

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
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # 5. PEFT / LoRA
    # ------------------------------------------------------------------
    if script_args.training_mode in ("lora", "qlora"):
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            target_modules=script_args.peft_target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
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
        eval_dataset=test_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
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

    # Save  (handles FSDP full-state-dict consolidation automatically)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.dataset_text_field is None:
        training_args.dataset_text_field = "text"
    if training_args.max_seq_length is None:
        training_args.max_seq_length = 2048
        
    training_args.packing = True
    training_args.dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token": False,
    }

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    set_seed(training_args.seed)

    training_function(script_args, training_args)
