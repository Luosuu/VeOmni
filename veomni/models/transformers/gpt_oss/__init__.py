# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ....lora.target_mapping import convert_fused_moe_lora_targets
from ...loader import MODELING_REGISTRY


def _convert_gpt_oss_lora_targets_to_parameters(_model, lora_modules, target_parameter_patterns):
    return convert_fused_moe_lora_targets(
        lora_modules,
        target_parameter_patterns,
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    )


@MODELING_REGISTRY.register("gpt_oss")
def register_gpt_oss_modeling(architecture: str):
    architecture = architecture or "GptOssForCausalLM"

    try:
        import transformers.models.gpt_oss
    except ImportError as e:
        raise RuntimeError(
            "GPT-OSS support requires a Transformers build that provides `transformers.models.gpt_oss`."
        ) from e
    from .generated.patched_modeling_gpt_oss_gpu import (
        GptOssForCausalLM,
        GptOssForSequenceClassification,
        GptOssForTokenClassification,
        GptOssModel,
    )

    for model_cls in (
        GptOssForCausalLM,
        GptOssForSequenceClassification,
        GptOssForTokenClassification,
    ):
        model_cls._convert_lora_targets_to_parameters = staticmethod(_convert_gpt_oss_lora_targets_to_parameters)

    if "ForSequenceClassification" in architecture:
        return GptOssForSequenceClassification
    elif "ForTokenClassification" in architecture:
        return GptOssForTokenClassification
    elif "Model" in architecture:
        return GptOssModel
    else:
        return GptOssForCausalLM
