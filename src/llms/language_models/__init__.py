from .chatgpt import ChatGPT
from .alpaca import Alpaca
from .longchat.longchat import Longchat
from .base_language_model import BaseLanguageModel
from .llama import Llama
from .flan_t5 import FlanT5
from .gemini import Gemini
from .local import LocalLLM

registed_language_models = {
    'gpt-4': ChatGPT,
    'gpt-3.5-turbo': ChatGPT,
    'alpaca': Alpaca,
    'longchat': Longchat,
    'llama': Llama,
    'flan-t5': FlanT5,
    'rog': Llama,
    'gemini-3.1-flash-lite': Gemini,
    'gemma-4-31b-it': Gemini,
    'local_model': LocalLLM,
    'kit.gpt-oss-120b': LocalLLM,
    'kit.mistral-small-4-119b-a8b': LocalLLM,
    'kit.minimax-m2.7-229b': LocalLLM,
    'kit.qwen3.5-397b-A17b': LocalLLM,
    'kit.gemma4-31b-it': LocalLLM,
    'kit.gpt-oss-120b-wo-tools': LocalLLM,
    'gpt-oss-120b-wo-tools': LocalLLM,
    'mistral-small-4-119b-a8b-wo-tools': LocalLLM,
    'qwen35-397b-a17b-wo-tools': LocalLLM,
    'minimax-m27-229b-wo-tools': LocalLLM,
    'gemma4-31b-it-wo-tools': LocalLLM
}

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        # if key in model_name.lower():
        if key in model_name:
            return value
    raise ValueError(f"No registered model found for name '{model_name}'")
