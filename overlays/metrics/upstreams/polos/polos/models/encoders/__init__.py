# -*- coding: utf-8 -*-
from .bert import BERTEncoder
from .encoder_base import Encoder

try:
    from .xlmr import XLMREncoder
except ModuleNotFoundError:
    XLMREncoder = None

str2encoder = {"BERT": BERTEncoder}
if XLMREncoder is not None:
    str2encoder["XLMR"] = XLMREncoder

__all__ = ["BERTEncoder", "XLMREncoder"]
