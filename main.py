# /usr/bin/env python3

import re
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
                description="Official FLUX.1-schnell model by BlackForestLabs"
            ),
            PickOption(
                "FLUX.1-dev (official)",
                value="black-forest-labs/FLUX.1-dev",
                description="Official FLUX.1-dev model by BlackForestLabs"
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
    pipeline.to(device)

    return (pipeline, config)


class HintCompleter(Completer):
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        flags = [
            "a",
            "again",
            "f",
            "fast",
            "quit",
            "r",
            "random",
            "s",
            "slow",
        ]
        if document.char_before_cursor == "/":
            for flag in flags:
                yield Completion(flag, start_position=-1 * len(flag))
        else:
            matches = re.match(r".*?/([a-z0-9]+)$", document.text_before_cursor)
            if matches:
                for flag in flags:
                    if flag.startswith(matches.group(1)):
                        full_flag = f"/{flag}"
                        yield Completion(full_flag, start_position=-1 * len(full_flag))


class FluxProompter:
    last_prompt: str
    last_seed: int
    hint_size: tuple[int, int]
    hint_inference_steps: int
    hint_reuse_seed: bool
    generator: torch.Generator
    pipeline: FluxPipeline
    config: FluxConfig
    session: PromptSession
    completions: Completer

    def __init__(self, pipeline: FluxPipeline, config: FluxConfig):
        self.last_prompt = ""
        self.last_seed = 0
        self.hint_size = (512, 512)
        self.hint_inference_steps = 4
        self.hint_reuse_seed = False
        self.generator = torch.Generator("cpu")
        self.pipeline = pipeline
        self.config = config
        self.session = PromptSession()
        self.completions = HintCompleter()

    def __bottom_toolbar(self):
        return HTML(
            f"<b>{self.config.model_name}</b> | Steps: {self.hint_inference_steps} | Size: {self.hint_size[0]}x{self.hint_size[1]} | Last Seed: {self.last_seed}"
        )

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
                print(f"-> Reusing last seed ({self.last_seed}).")
                self.hint_reuse_seed = True
            elif hint == "r" or hint == "random":
                print("-> Using random seed.")
                self.hint_reuse_seed = False
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

        # Check for quitters
        if user_prompt == "/quit":
            exit(0)

        # Parse hints and get final prompt with hints removed
        user_prompt = self.__parse_hints(user_prompt)

        # Reuse last prompt if we're left with nothing
        if user_prompt.strip() == "":
            print(f"Reusing last prompt: {self.last_prompt}")
            user_prompt = self.last_prompt

        # Seed shenanigans
        if self.hint_reuse_seed:
            print(f"Reusing last seed: {self.last_seed}")
        seed = self.generator.seed() if not self.hint_reuse_seed else self.last_seed
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

        # Obtain final image and save it
        image = result.images[0]
        image.save(f"output/flux_{time.time()}_{seed}.png")

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
