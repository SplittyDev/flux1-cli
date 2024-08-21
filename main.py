# /usr/bin/env python3

import re
import os
import time
import argparse
from typing import Iterable
from prompt_toolkit.completion.base import CompleteEvent
from prompt_toolkit.document import Document
import torch
import diffusers
from diffusers import FluxPipeline
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion
from pick import pick, Option as PickOption


def fix_flux_rope_transformer():
    """
    Fix the transformer used in the flux model to work with MPS.
    Does nothing if the pipeline is not using MPS.
    """

    _flux_rope = diffusers.models.transformers.transformer_flux.rope

    def patched_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0
        if pos.device.type == "mps":
            return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
        return _flux_rope(pos, dim, theta)

    diffusers.models.transformers.transformer_flux.rope = patched_flux_rope


class AppConfig:
    """
    Configuration for the application.
    """

    offload_cpu: bool
    force_cpu: bool

    def __init__(self, offload_cpu: bool, force_cpu: bool):
        self.offload_cpu = offload_cpu
        self.force_cpu = force_cpu


class FluxConfig:
    """
    Configuration for model-specific inference and UI.
    """

    model_name: str
    fast_iteration_count: int
    quality_iteration_count: int
    fixed_guidance_scale: bool
    default_guidance_scale: float
    max_sequence_length: int
    torch_dtype: torch.dtype

    def __init__(
        self,
        model_name: str,
        fast_iteration_count: int,
        quality_iteration_count: int,
        fixed_guidance_scale: bool,
        default_guidance_scale: float,
        max_sequence_length: int,
        torch_dtype: torch.dtype,
    ):
        self.model_name = model_name
        self.fast_iteration_count = fast_iteration_count
        self.quality_iteration_count = quality_iteration_count
        self.fixed_guidance_scale = fixed_guidance_scale
        self.default_guidance_scale = default_guidance_scale
        self.max_sequence_length = max_sequence_length
        self.torch_dtype = torch_dtype


def get_best_device(force_cpu: bool = False) -> torch.device:
    """
    Attempt to find the fastest device for inference.
    """

    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend (GPU).")
    elif not force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend (GPU).")
    else:
        device = torch.device("cpu")
        if force_cpu:
            print("Using CPU backend.")
        else:
            print("No GPU detected. Using CPU as fallback.")
    return device


def load_model_interactive(device: torch.device) -> tuple[FluxPipeline, FluxConfig]:
    # Show interactive model picker
    model_option = pick(
        [
            PickOption(
                "FLUX.1-schnell (official)",
                value="black-forest-labs/FLUX.1-schnell",
                description="Official FLUX.1-schnell model by BlackForestLabs",
            ),
            PickOption(
                "FLUX.1-dev (official)",
                value="black-forest-labs/FLUX.1-dev",
                description="Official FLUX.1-dev model by BlackForestLabs",
            ),
        ],
        title="Choose a model",
    )[0]

    model_repo = model_option.value
    model_name = model_option.value.split("/")[1]

    # Instantiate model-specific configuration
    config: FluxConfig
    match model_name:
        case "FLUX.1-schnell":
            config = FluxConfig(
                model_name=model_name,
                fast_iteration_count=1,
                quality_iteration_count=4,
                fixed_guidance_scale=True,
                default_guidance_scale=0,
                max_sequence_length=256,
                torch_dtype=torch.bfloat16,
            )
        case "FLUX.1-dev":
            config = FluxConfig(
                model_name=model_name,
                fast_iteration_count=5,
                quality_iteration_count=28,
                fixed_guidance_scale=False,
                default_guidance_scale=3.5,
                max_sequence_length=512,
                torch_dtype=torch.bfloat16,
            )

    # Load inference pipeline
    pipeline: FluxPipeline = FluxPipeline.from_pretrained(
        model_repo, torch_dtype=config.torch_dtype
    )

    # Move pipeline to device
    pipeline.to(device)

    return (pipeline, config)


class HintCompleter(Completer):
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        suffix_flags = [
            "a",
            "again",
            "f",
            "fast",
            "r",
            "random",
            "s",
            "slow",
            "seed=",
        ]
        standalone_flags = [
            "quit",
            "seed=",
            "lora",
        ]
        if document.char_before_cursor == "/":
            if document.cursor_position_col == 1:
                for flag in standalone_flags:
                    yield Completion(flag, start_position=-1 * len(flag))
            for flag in suffix_flags:
                yield Completion(flag, start_position=-1 * len(flag))
        else:
            matches = re.match(r".*?/([a-z0-9]+)$", document.text_before_cursor)
            if matches:
                if document.text.strip().startswith("/"):
                    for flag in standalone_flags:
                        if flag.startswith(matches.group(1)):
                            full_flag = f"/{flag}"
                            yield Completion(flag, start_position=-1 * len(full_flag))
                for flag in suffix_flags:
                    if flag.startswith(matches.group(1)):
                        full_flag = f"/{flag}"
                        yield Completion(full_flag, start_position=-1 * len(full_flag))


class FluxProompter:
    last_prompt: str
    last_seed: int
    fixed_seed: int | None
    hint_size: tuple[int, int]
    hint_inference_steps: int
    generator: torch.Generator
    pipeline: FluxPipeline
    config: FluxConfig
    session: PromptSession
    completions: Completer

    def __init__(self, pipeline: FluxPipeline, config: FluxConfig):
        self.last_prompt = ""
        self.last_seed = 0
        self.fixed_seed = None
        self.hint_size = (512, 512)
        self.hint_inference_steps = 4
        self.generator = torch.Generator("cpu")
        self.pipeline = pipeline
        self.config = config
        self.session = PromptSession()
        self.completions = HintCompleter()

    def __bottom_toolbar(self):
        items: list[str] = []
        model_name = f"<b>{self.config.model_name}</b>"
        step_count = f"Steps: {self.hint_inference_steps}"
        resolution = f"Resolution: {self.hint_size[0]}x{self.hint_size[1]}"
        items.append(model_name)
        items.append(step_count)
        items.append(resolution)
        if self.fixed_seed is None:
            items.append("Seed: Random")
            items.append(f"Last Seed: {self.last_seed}")
        else:
            items.append(f"Seed: Fixed ({self.fixed_seed})")
        item_code = " | ".join(items)
        return HTML(item_code)

    def __show_lora_picker(self):
        available_loras = [f for f in os.listdir("lora/") if f.endswith(".safetensors")]
        if len(available_loras) == 0:
            print(
                "No LORAs found. Place the .safetensors files in the lora/ directory."
            )
            return
        selected_loras = pick(
            available_loras, title="Choose LoRA weights to load", multiselect=True
        )
        self.pipeline.unload_lora_weights()
        for path, _ in selected_loras:
            filename = os.path.basename(path)
            self.pipeline.load_lora_weights(f"./lora/{filename}")
            print(f"-> Loaded LoRA: {filename}")

    def __parse_hints(self, user_prompt: str) -> str:
        hints = user_prompt.split("/")
        for hint in hints[1:]:
            if hint.isdigit():
                num = int(hint)
                if num > 128:
                    print(f"-> Using {num}x{num} resolution.")
                    self.hint_size = (num, num)
                else:
                    print(f"-> Using {num} inference steps.")
                    self.hint_inference_steps = num
            elif "x" in hint:
                [w, h] = hint.split("x")
                assert w.isdigit(), "The width has to be a number!"
                assert h.isdigit(), "The height has to be a number!"
                print(f"-> Using {w}x{h} resolution.")
                self.hint_size = (int(w), int(h))
            elif hint == "f" or hint == "fast":
                print("-> Using fast generation.")
                self.hint_inference_steps = self.config.fast_iteration_count
            elif hint == "s" or hint == "slow":
                print("-> Using high quality generation.")
                self.hint_inference_steps = self.config.quality_iteration_count
            elif hint == "a" or hint == "again" or hint == "=":
                print(f"-> Using fixed seed ({self.last_seed}).")
                self.fixed_seed = self.last_seed
            elif hint == "r" or hint == "random":
                print("-> Using random seed.")
                self.fixed_seed = None
            elif hint.startswith("seed="):
                seed = int(hint.split("=")[1])
                print(f"-> Using fixed seed ({seed}).")
                self.fixed_seed = seed
        return hints[0]

    def run(self):
        print("Type /quit to quit properly, or press Ctrl+C to quit like a madman.")

        while True:
            self.run_once()

    def run_once(self):
        # Show prompt prompt (yes, that's not a typo)
        user_prompt = self.session.prompt(
            "λ Prompt: ",
            completer=self.completions,
            bottom_toolbar=self.__bottom_toolbar,
        ).strip()
        prompt_started_with_hint = user_prompt.startswith("/")

        # Check standalone hints
        match user_prompt:
            case "/quit":
                exit(0)
            case "/lora":
                self.__show_lora_picker()
                return
            case hint if re.match(r"/seed\s*=\s*\d+", hint) is not None:
                seed = int(hint.split("=")[1].strip())
                self.fixed_seed = seed
                print(f"-> Using fixed seed ({seed}).")
                return

        # Parse suffix hints and get final prompt with hints removed
        user_prompt = self.__parse_hints(user_prompt)

        # Do not run inference on prompts that start with a hint,
        # unless there is a cached prompt to reuse.
        if prompt_started_with_hint and self.last_prompt.strip() == "":
            return

        # Reuse last prompt if we're left with nothing
        if user_prompt.strip() == "":
            print(f"-> Reusing last prompt ({self.last_prompt})")
            user_prompt = self.last_prompt

        # Seed shenanigans
        seed = self.generator.seed() if self.fixed_seed is None else self.fixed_seed
        generator = self.generator.manual_seed(seed)
        self.last_seed = seed

        # Party time
        result = self.pipeline(
            user_prompt,
            width=self.hint_size[0],
            height=self.hint_size[1],
            guidance_scale=self.config.default_guidance_scale,
            num_inference_steps=self.hint_inference_steps,
            max_sequence_length=self.config.max_sequence_length,
            generator=generator,
        )

        # Create folder for current date
        date = time.strftime("%Y-%m-%d")
        os.makedirs(f"output/{date}", exist_ok=True)

        # Obtain final image and save it
        image = result.images[0]
        image.save(f"output/{date}/{int(time.time())}_{seed}.png")

        # Update last prompt
        self.last_prompt = user_prompt


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Run FLUX.1 models interactively.")
    parser.add_argument("--offload-cpu", action="store_true", help="Use less VRAM.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()
    return AppConfig(offload_cpu=args.offload_cpu, force_cpu=args.force_cpu)


def main():
    # Fix transformer for MPS
    fix_flux_rope_transformer()

    # Parse command line arguments
    app_config = parse_args()

    # This should be self-explanatory
    device = get_best_device(force_cpu=app_config.force_cpu)

    # Load model config and pipeline
    pipeline, model_config = load_model_interactive(device=device)

    # Offload to CPU if requested
    if app_config.offload_cpu:
        if device.type == "mps":
            print("Error: Cannot offload to CPU when using MPS.")
            exit(1)
        else:
            pipeline.enable_model_cpu_offload()

    # Start inference loop
    FluxProompter(pipeline, model_config).run()


if __name__ == "__main__":
    main()
