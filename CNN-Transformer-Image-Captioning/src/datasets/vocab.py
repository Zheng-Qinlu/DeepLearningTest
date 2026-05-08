"""
Tokenizer/Vocab 封装（基于 HuggingFace Transformers 的 BertTokenizer）。

说明：
- 按你的要求，采用预训练好的 BERT 分词器作为文本处理方案。
- BOS/ EOS/ PAD/ UNK 分别对应 [CLS]/[SEP]/[PAD]/[UNK]。
- 复杂分词细节由 transformers 库处理。

参考实现：
- HuggingFace Transformers 文档与示例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
	from transformers import BertTokenizer
except Exception as e:  # pragma: no cover
	BertTokenizer = None  # 延迟导入错误到实际构造时提示


@dataclass
class TokenizerConfig:
	name: str = "bert-base-uncased"
	do_lower_case: bool = True


class BertTokenizerWrapper:
	"""
	一个轻量封装，统一我们项目中的 tokenizer 接口。
	"""

	def __init__(self, cfg: TokenizerConfig | Dict[str, Any] = TokenizerConfig()):
		if isinstance(cfg, dict):
			cfg = TokenizerConfig(**cfg)
		if BertTokenizer is None:
			raise ImportError("transformers 未安装，无法使用 BertTokenizer。请先 pip install transformers")
		self.tokenizer = BertTokenizer.from_pretrained(cfg.name, do_lower_case=cfg.do_lower_case)

		# special tokens
		self.pad_token = self.tokenizer.pad_token or "[PAD]"
		self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.convert_tokens_to_ids(self.pad_token)

		self.bos_token = self.tokenizer.cls_token or "[CLS]"
		self.bos_token_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.convert_tokens_to_ids(self.bos_token)

		self.eos_token = self.tokenizer.sep_token or "[SEP]"
		self.eos_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.convert_tokens_to_ids(self.eos_token)

		self.unk_token = self.tokenizer.unk_token or "[UNK]"
		self.unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else self.tokenizer.convert_tokens_to_ids(self.unk_token)

	@property
	def vocab_size(self) -> int:
		return self.tokenizer.vocab_size

	def tokenize(self, text: str) -> List[str]:
		return self.tokenizer.tokenize(text)

	def encode(
		self,
		text: str,
		add_special_tokens: bool = True,
		max_length: Optional[int] = None,
		truncation: bool = True,
	) -> List[int]:
		return self.tokenizer.encode(
			text,
			add_special_tokens=add_special_tokens,
			max_length=max_length,
			truncation=truncation,
		)

	def batch_encode(self, texts: List[str], add_special_tokens: bool = True, max_length: Optional[int] = None,
					 truncation: bool = True) -> List[List[int]]:
		return [self.encode(t, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation) for t in texts]

	def pad_sequences(self, sequences: List[List[int]], max_len: Optional[int] = None) -> List[List[int]]:
		if max_len is None:
			max_len = max(len(s) for s in sequences) if sequences else 0
		padded: List[List[int]] = []
		for s in sequences:
			if len(s) < max_len:
				padded.append(s + [self.pad_token_id] * (max_len - len(s)))
			else:
				padded.append(s[:max_len])
		return padded

	def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
		return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

	def save_pretrained(self, save_directory: str):
		self.tokenizer.save_pretrained(save_directory)


__all__ = [
	"TokenizerConfig",
	"BertTokenizerWrapper",
]
