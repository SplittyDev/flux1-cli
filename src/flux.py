import os
import re
import time
import torch
from pick import pick
from typing import Iterable
from diffusers import FluxPipeline
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.completion.base import CompleteEvent
from prompt_toolkit.document import Document

from flux_config import FluxConfig


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
        ]
        standalone_flags = ["quit", "seed=", "lora", "lora_scale="]
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
    lora_scale: float
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
        self.lora_scale = 0.5
        self.generator = torch.Generator("cpu")
        self.pipeline = pipeline
        self.config = config
        self.session = PromptSession()
        self.completions = HintCompleter()

    def __bottom_toolbar(self):
        items: list[str] = []
        model_name = f"<b>{self.config.model_display_name}</b>"
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
        for hint in [s.strip() for s in hints[1:]]:
            if hint.isdigit():
                num = int(hint)
                if num > 128:
                    print(f"-> Using {num}x{num} resolution.")
                    self.hint_size = (num, num)
                else:
                    print(f"-> Using {num} inference steps.")
                    self.hint_inference_steps = num
            elif "x" in hint:
                [w, h] = [s.strip() for s in hint.split("x")]
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
            case hint if re.match(
                r"/lora_scale\s*=\s*(\d+(?:\.\d+)?)", hint
            ) is not None:
                scale = float(hint.split("=")[1].strip())
                self.lora_scale = scale
                print(f"-> Set LoRA scale to {scale}.")
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
            joint_attention_kwargs={"scale": self.lora_scale},
        )

        # Create folder for current date
        date = time.strftime("%Y-%m-%d")
        os.makedirs(f"output/{date}", exist_ok=True)

        # Obtain final image and save it
        image = result.images[0]
        image.save(f"output/{date}/{int(time.time())}_{seed}.png")
        image.save("output/latest.png")

        # Update last prompt
        self.last_prompt = user_prompt
