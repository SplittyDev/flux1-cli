import torch


class FluxConfig:
    """
    Configuration for model-specific inference and UI.
    """

    repo: str
    model_display_name: str
    model_safetensors: str | None
    fast_iteration_count: int
    quality_iteration_count: int
    fixed_guidance_scale: bool
    default_guidance_scale: float
    max_sequence_length: int
    torch_dtype: torch.dtype

    def __init__(
        self,
        repo: str,
        model_display_name: str,
        model_safetensors: str | None,
        fast_iteration_count: int,
        quality_iteration_count: int,
        fixed_guidance_scale: bool,
        default_guidance_scale: float,
        max_sequence_length: int,
        torch_dtype: torch.dtype,
    ):
        self.repo = repo
        self.model_display_name = model_display_name
        self.model_safetensors = model_safetensors
        self.fast_iteration_count = fast_iteration_count
        self.quality_iteration_count = quality_iteration_count
        self.fixed_guidance_scale = fixed_guidance_scale
        self.default_guidance_scale = default_guidance_scale
        self.max_sequence_length = max_sequence_length
        self.torch_dtype = torch_dtype
