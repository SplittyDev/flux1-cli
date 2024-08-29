# /usr/bin/env python3

import torch
import argparse
import diffusers

from diffusers import FluxPipeline
from pick import pick, Option as PickOption

from app_config import AppConfig
from flux_config import FluxConfig
from torch_helper import get_best_device
from flux import FluxProompter
from flux_pipeline import FullFluxLoader


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


def load_model_interactive(device: torch.device) -> tuple[FluxPipeline, FluxConfig]:
    # Show interactive model picker
    model_option = pick(
        [
            PickOption(
                "FLUX.1-schnell (official)",
                value="flux1-schnell",
                description="FLUX.1-schnell by BlackForestLabs",
            ),
            PickOption(
                "FLUX.1-dev (official)",
                value="flux1-dev",
                description="FLUX.1-dev by BlackForestLabs",
            ),
            PickOption(
                "FLUX.1-schnell (community, FP8)",
                value="flux1-schnell-fp8",
                description="FP8 quantized checkpoint",
            ),
            PickOption(
                "FLUX.1-dev (community, FP8)",
                value="flux1-dev-fp8",
                description="FP8 quantized checkpoint",
            ),
        ],
        title="Choose a model",
    )[0]

    # Instantiate model-specific configuration
    config: FluxConfig
    match model_option.value:
        case "flux1-schnell":
            config = FluxConfig(
                repo="black-forest-labs/FLUX.1-schnell",
                model_display_name="flux1-schnell",
                model_safetensors=None,
                fast_iteration_count=1,
                quality_iteration_count=4,
                fixed_guidance_scale=True,
                default_guidance_scale=0,
                max_sequence_length=256,
                torch_dtype=torch.bfloat16,
            )
        case "flux1-dev":
            config = FluxConfig(
                repo="black-forest-labs/FLUX.1-dev",
                model_display_name="flux1-dev",
                model_safetensors=None,
                fast_iteration_count=5,
                quality_iteration_count=28,
                fixed_guidance_scale=False,
                default_guidance_scale=3.5,
                max_sequence_length=512,
                torch_dtype=torch.bfloat16,
            )
        case "flux1-schnell-fp8":
            config = FluxConfig(
                repo="kijai/flux-fp8",
                model_display_name="kijai-schnell-fp8",
                model_safetensors="flux1-schnell-fp8.safetensors",
                fast_iteration_count=1,
                quality_iteration_count=4,
                fixed_guidance_scale=True,
                default_guidance_scale=0,
                max_sequence_length=256,
                torch_dtype=torch.float8_e4m3fn,
            )
        case "flux1-dev-fp8":
            config = FluxConfig(
                repo="kijai/flux-fp8",
                model_safetensors="flux1-dev-fp8.safetensors",
                model_display_name="kijai-dev-fp8",
                fast_iteration_count=5,
                quality_iteration_count=28,
                fixed_guidance_scale=False,
                default_guidance_scale=3.5,
                max_sequence_length=512,
                torch_dtype=torch.float8_e4m3fn,
            )

    # Load inference pipeline
    pipeline: FluxPipeline = FullFluxLoader(device, config).load()

    # Move pipeline to device
    pipeline.to(device)

    return (pipeline, config)


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Run FLUX.1 models interactively.")
    parser.add_argument("--offload-cpu", action="store_true", help="Use less VRAM.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()
    return AppConfig(offload_cpu=args.offload_cpu, force_cpu=args.force_cpu)


@torch.inference_mode()
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
