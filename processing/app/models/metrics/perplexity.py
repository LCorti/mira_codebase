from typing import Any

import evaluate
import torch

from app.models.metrics.base_metric import Metric
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="metrics.log")


class PerplexityMetric(Metric):
    def __init__(self):
        """
        Initialize the PerplexityMetric.
        """
        super().__init__(
            name="perplexity",
            description="Computes the perplexity metric for model predictions.",
        )
        self.metric = evaluate.load("perplexity", module_type="metric")

    def compute(
        self,
        predictions: list[str],
        batch_size: int = 16,
        add_start_token: bool = True,
        device: str = "auto",
        **kwargs: Any,
    ) -> dict:
        """
        Compute the Perplexity metric for the given predictions using a specified model.
        See more details about Perplexity at: https://huggingface.co/spaces/evaluate-metric/perplexity

        :param predictions: List of input text, where each separate text snippet is one list entry.
        :type predictions: List[str]
        :param batch_size: The batch size to run texts through the model. Defaults to 16.
        :type batch_size: int, optional
        :param add_start_token: Whether to add the start token to the texts, so the perplexity can include the probability of the first word. Defaults to True.
        :type add_start_token: bool, optional
        :param device: Device to run on, defaults to cuda when available.
        :type device: str, optional
        :param kwargs: Additional keyword arguments for metric computation.
        :type kwargs: Any

        :return: Dictionary with 'perplexities' (list of individual perplexity scores) and 'mean_perplexity' (average perplexity).
        :rtype: dict
        """
        # Validate inputs
        if not predictions:
            raise ValueError("Predictions must not be empty.")
        if batch_size <= 0 or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer.")

        from app.config.registry import get_model_loader

        model_id = get_model_loader().get_model_name()

        device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            result = self.metric.compute(
                predictions=predictions,
                model_id=model_id,
                batch_size=batch_size,
                add_start_token=add_start_token,
                device=device,
                **kwargs,
            )

            return {"average perplexity": result.get("perplexities", [])}
        except Exception as e:
            # Check if exception starts with "Unrecognized"
            message = f"Error computing {self.name} metric: {e!s}"
            if message.startswith("Unrecognized"):
                message += " Model type should be one of ArceeConfig, AriaTextConfig, BambaConfig, BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig, BitNetConfig, BlenderbotConfig, BlenderbotSmallConfig, BloomConfig, CamembertConfig, LlamaConfig, CodeGenConfig, CohereConfig, Cohere2Config, CpmAntConfig, CTRLConfig, Data2VecTextConfig, DbrxConfig, DeepseekV2Config, DeepseekV3Config, DiffLlamaConfig, DogeConfig, Dots1Config, ElectraConfig, Emu3Config, ErnieConfig, Ernie4_5Config, Ernie4_5_MoeConfig, Exaone4Config, FalconConfig, FalconH1Config, FalconMambaConfig, FuyuConfig, GemmaConfig, Gemma2Config, Gemma3Config, Gemma3TextConfig, Gemma3nConfig, Gemma3nTextConfig, GitConfig, GlmConfig, Glm4Config, Glm4MoeConfig, GotOcr2Config, GPT2Config, GPT2Config, GPTBigCodeConfig, GPTNeoConfig, GPTNeoXConfig, GPTNeoXJapaneseConfig, GPTJConfig, GraniteConfig, GraniteMoeConfig, GraniteMoeHybridConfig, GraniteMoeSharedConfig, HeliumConfig, JambaConfig, JetMoeConfig, Lfm2Config, LlamaConfig, Llama4Config, Llama4TextConfig, MambaConfig, Mamba2Config, MarianConfig, MBartConfig, MegaConfig, MegatronBertConfig, MiniMaxConfig, MistralConfig, MixtralConfig, MllamaConfig, ModernBertDecoderConfig, MoshiConfig, MptConfig, MusicgenConfig, MusicgenMelodyConfig, MvpConfig, NemotronConfig, OlmoConfig, Olmo2Config, OlmoeConfig, OpenLlamaConfig, OpenAIGPTConfig, OPTConfig, PegasusConfig, PersimmonConfig, PhiConfig, Phi3Config, Phi4MultimodalConfig, PhimoeConfig, PLBartConfig, ProphetNetConfig, QDQBertConfig, Qwen2Config, Qwen2MoeConfig, Qwen3Config, Qwen3MoeConfig, RecurrentGemmaConfig, ReformerConfig, RemBertConfig, RobertaConfig, RobertaPreLayerNormConfig, RoCBertConfig, RoFormerConfig, RwkvConfig, SmolLM3Config, Speech2Text2Config, StableLmConfig, Starcoder2Config, TransfoXLConfig, TrOCRConfig, WhisperConfig, XGLMConfig, XLMConfig, XLMProphetNetConfig, XLMRobertaConfig, XLMRobertaXLConfig, XLNetConfig, xLSTMConfig, XmodConfig, ZambaConfig, Zamba2Config."
            raise RuntimeError(f"Failed to compute {self.name} metric: {e!s}") from e
