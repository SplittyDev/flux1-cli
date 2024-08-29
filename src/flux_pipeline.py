import torch

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel

from flux_config import FluxConfig

bfl_repo = "black-forest-labs/FLUX.1-dev"


class FullFluxLoader:
    device: torch.device
    config: FluxConfig

    def __init__(self, device: torch.device, config: FluxConfig):
        self.device = device
        self.config = config

    def _load_transformer(self) -> FluxTransformer2DModel:
        transformer_url = f"https://huggingface.co/{self.config.repo}/blob/main/{self.config.model_safetensors}"
        transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_single_file(
            transformer_url,
            # config=self.config.repo,
            torch_dtype=self.config.torch_dtype,
        )
        return transformer.to(self.device)

    def _load_t5_encoder(self) -> T5EncoderModel:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        return text_encoder_2.to(self.device)

    def load(self) -> FluxPipeline:
        if self.config.model_safetensors is None:
            pipe = FluxPipeline.from_pretrained(
                bfl_repo, torch_dtype=torch.bfloat16
            )
        else:
            transformer = self._load_transformer()
            text_encoder_2 = self._load_t5_encoder()
            pipe = FluxPipeline.from_pretrained(
                bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=torch.bfloat16
            )
            pipe.transformer = transformer
            pipe.text_encoder_2 = text_encoder_2
        return pipe
