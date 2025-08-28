import logging
from typing import Any

from transformers import PretrainedConfig
from transformers.models.biogpt.configuration_biogpt import (
    BioGptConfig as _BioGptConfig,
)

logger = logging.getLogger(__name__)


class BioGptConfig(_BioGptConfig):
    base_config_key = "biogpt_config"


class PerceiverConfig(PretrainedConfig):
    model_type = "perceiver"
    base_config_key = "perceiver_config"

    def __init__(
        self,
        latent_seq: int = 512,
        latent_dim: int = 1280,
        context_dim: int = 2560,
        mhsa_heads: int = 8,
        perceiver_depth: int = 8,
        transformer_depth: int = 6,
        share_xattn_start_layer: int = 1,
        share_tf_start_layer: int = 0,
        xattn_heads: int = 1,
        mlp_mult: int = 1,
        mlp_activation: str = 'geglu',
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.latent_seq = latent_seq
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.mhsa_heads = mhsa_heads
        self.perceiver_depth = perceiver_depth
        self.transformer_depth = transformer_depth
        self.share_xattn_start_layer = share_xattn_start_layer
        self.share_tf_start_layer = share_tf_start_layer
        self.xattn_heads = xattn_heads
        self.mlp_mult = mlp_mult
        self.mlp_activation = mlp_activation


class PrismConfig(PretrainedConfig):
    model_type = "prism"
    sub_configs = {"biogpt_config": BioGptConfig, "perceiver_config": PerceiverConfig}

    def __init__(
        self,
        biogpt_config: dict | BioGptConfig | None = None,
        perceiver_config: dict | PerceiverConfig | None = None,
        biogpt_context_dim: int = 1280,
        biogpt_frozen_weights: bool = True,
        biogpt_frozen_embeddings: bool = False,
        dim_latents: int = 5120,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if isinstance(biogpt_config, dict):
            self.biogpt_config = BioGptConfig(**biogpt_config)
        elif isinstance(biogpt_config, BioGptConfig):
            self.biogpt_config = biogpt_config
        elif biogpt_config is None:
            self.biogpt_config = BioGptConfig()
            logger.info(
                "`biogpt_config` is `None`. Initializing the `BioGptConfig` with default values."
            )
        else:
            raise ValueError(
                "`biogpt_config` must be a `dict`, `BioGptConfig` instance, or `None`."
            )

        if isinstance(perceiver_config, dict):
            self.perceiver_config = PerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, PerceiverConfig):
            self.perceiver_config = perceiver_config
        elif perceiver_config is None:
            self.perceiver_config = PerceiverConfig()
            logger.info(
                "`perceiver_config` is `None`. Initializing the `PerceiverConfig` with default values."
            )
        else:
            raise ValueError(
                "`perceiver_config` must be a `dict`, `PerceiverConfig` instance, or `None`."
            )

        self.biogpt_context_dim = biogpt_context_dim
        self.biogpt_frozen_weights = biogpt_frozen_weights
        self.biogpt_frozen_embeddings = biogpt_frozen_embeddings

        self.dim_latents = dim_latents

        super().__init__(**kwargs)

