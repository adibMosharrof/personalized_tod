from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from dotmap import DotMap

from transformers import AutoTokenizer, GPT2LMHeadModel

from tokenizers.processors import TemplateProcessing


class SpecialTokens(str, Enum):
    begin_target = "<|begin_target|>"
    end_target = "<|end_target|>"

    begin_context = "<|begin_context|>"
    end_context = "<|end_context|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"

    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"

    begin_query = "<|beginquery|>"
    end_query = "<|endquery|>"

    begin_slots = "<|beginslots|>"
    end_slots = "<|endslots|>"

    silence = "<SILENCE>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class SpecialPredictions(str, Enum):
    DUMMY = "DUMMY"


class TokenizerTokens(str, Enum):
    pad_token = "<|pad|>"
    eos_token = "<|endoftext|>"
    bos_token = "<|startoftext|>"


Steps = DotMap(
    train="trn",
    val="dev",
    test="tst",
)


class TodUtils:
    @classmethod
    def get_tokenizer(self, model_name: str = "distilgpt2") -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            truncation_side="left",
            pad_token=TokenizerTokens.pad_token.value,
            bos_token=TokenizerTokens.bos_token.value,
            eos_token=TokenizerTokens.eos_token.value,
            additional_special_tokens=SpecialTokens.list(),
        )

        # special_tokens = SpecialTokens.list()
        # tokenizer.add_tokens(special_tokens, special_tokens=True)
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0",
            special_tokens=[
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id),
            ],
        )
        return tokenizer

    @classmethod
    def load_model_tokenizer(
        self, model_path: str, project_root: Path
    ) -> Optional[Tuple[GPT2LMHeadModel, AutoTokenizer]]:
        if not model_path:
            return None, None
        model_path = project_root / model_path
        m = GPT2LMHeadModel.from_pretrained(model_path).cuda()
        t = AutoTokenizer.from_pretrained(model_path.parent.parent)
        return m, t
