# Copyright 2026 Google LLC
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

"""Transformers registration shim for ``model_type="qwen4_exp"``.

Diagnosed 2026-09-16 (Kaggle v44): the vLLM server dies in
``ModelConfig`` / ``repo_utils.get_config`` with::

    ValueError: The checkpoint you are trying to load has model type
    `qwen4_exp` but Transformers does not recognize this architecture.

even on transformers 5.12.1, because stock transformers has no
``qwen4_exp`` entry in ``CONFIG_MAPPING``. vLLM resolves the arch to our
JAX implementation only *after* parsing the HF config, so the parse must
succeed first.

The fix: register a ``Qwen4ExpConfig`` with ``AutoConfig`` before vLLM
touches the checkpoint. The text trunk derives from Qwen3-Next, so the
text config subclasses transformers' ``Qwen3NextConfig`` (all standard
fields vLLM reads keep working); qwen4_exp extras (``hc_*``, ``ple_*``,
``indexer_*``, ``partial_rotary_factor``) flow through ``**kwargs`` and
are preserved as plain attributes.

Entrypoints:
- :func:`install_hf_config` — idempotent, safe to call repeatedly and
  from any process (notebook cell, vLLM server subprocess).
- ``startup.py`` calls it on import; the fork cell drops a ``.pth`` so
  every fresh interpreter (notably the ``vllm serve`` subprocess)
  self-installs before ``ModelConfig`` runs.
"""

from __future__ import annotations

from typing import Any

MODEL_TYPE = "qwen4_exp"

# Class cache so repeated installs reuse the same objects.
_BUILT: tuple[Any, Any] | None = None


def _strip_quantization_config(cfg_obj: Any) -> None:
    """Best-effort post-init removal (from_dict strip is the real guard)."""
    try:
        del cfg_obj.__dict__["quantization_config"]
    except (KeyError, AttributeError, TypeError):
        pass


def _build_config_classes() -> tuple[Any, Any]:
    """Create (Qwen4ExpTextConfig, Qwen4ExpConfig). Import-lazy on purpose:
    this package must stay importable without transformers (CPU unit tests,
    JAX-only paths)."""
    global _BUILT
    if _BUILT is not None:
        return _BUILT
    from transformers import PretrainedConfig

    try:
        from transformers import Qwen3NextConfig as _TextBase
    except ImportError:  # very old transformers; minimal but functional
        _TextBase = PretrainedConfig

    class Qwen4ExpTextConfig(_TextBase):  # type: ignore[valid-type,misc]
        model_type = MODEL_TYPE

    class Qwen4ExpConfig(PretrainedConfig):
        """Top-level wrapper: mirrors the checkpoint layout where the text
        trunk lives under ``text_config`` (falls back to flat)."""

        model_type = MODEL_TYPE

        def __init__(self, text_config: Any = None, **kwargs: Any) -> None:
            if isinstance(text_config, dict):
                text_config = Qwen4ExpTextConfig(**text_config)
            # Set BEFORE super().__init__: transformers>=5.12 dataclass-validates
            # inside __init__ (validate_token_ids -> get_text_config), so the
            # attribute must already exist (v46 died here).
            self.text_config = text_config
            super().__init__(**kwargs)
            # Belt-and-suspenders: from_dict (below) already strips
            # quantization_config before construction; this covers direct
            # construction paths.
            _strip_quantization_config(self)
            if isinstance(text_config, PretrainedConfig):
                _strip_quantization_config(text_config)

        @classmethod
        def from_dict(cls, config_dict: Any, **kwargs: Any) -> Any:
            """Strip quantization_config BEFORE construction (airtight).

            vLLM resolves its (CUDA-only) quant path from the checkpoint's
            quantization_config via override hooks (v48 GPTQ gate, v51
            modelopt_fp4 JAX-universe gate). We serve quantized weights via
            JAX-side load-time dequant, so the entry must never reach the
            parsed config. Stripping here (rather than only post-init)
            is immune to validated-setattr/delattr quirks: v51 died because
            a stash setattr raised inside try/except, silently skipping the
            delattr on transformers 5.12.
            """
            if isinstance(config_dict, dict):
                config_dict = dict(config_dict)
                qc = config_dict.pop("quantization_config", None)
                text = config_dict.get("text_config")
                if isinstance(text, dict):
                    text = dict(text)
                    text_qc = text.pop("quantization_config", None)
                    config_dict["text_config"] = text
                    qc = qc if qc is not None else text_qc
                cfg = super().from_dict(config_dict, **kwargs)
                if qc is not None:
                    try:
                        object.__setattr__(
                            cfg, "qwen4exp_quantization_config", qc)
                    except Exception:
                        pass
                return cfg
            return super().from_dict(config_dict, **kwargs)

        def get_text_config(self, *args: Any, **kwargs: Any) -> Any:
            # __dict__ lookup: recursion-proof against the custom
            # __getattribute__ transformers>=5.12 installs on configs.
            tc = self.__dict__.get("text_config", None)
            if tc is not None:
                return tc
            return super().get_text_config(*args, **kwargs)

    _BUILT = (Qwen4ExpTextConfig, Qwen4ExpConfig)
    return _BUILT


def install_hf_config() -> Any:
    """Register ``qwen4_exp`` with transformers' ``AutoConfig``.

    Returns the registered ``Qwen4ExpConfig`` class. Idempotent: safe to
    call on every run / in every process.
    """
    _, config_cls = _build_config_classes()
    from transformers import AutoConfig

    try:
        AutoConfig.register(MODEL_TYPE, config_cls, exist_ok=True)
    except TypeError:  # ancient transformers without exist_ok
        try:
            AutoConfig.register(MODEL_TYPE, config_cls)
        except ValueError:
            pass  # already registered
    except ValueError:
        pass  # already registered
    return config_cls


__all__ = ["MODEL_TYPE", "install_hf_config"]
