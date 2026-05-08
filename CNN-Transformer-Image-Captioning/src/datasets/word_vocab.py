"""
真实词汇表构建（直接参考/改写自 MIT 许可证仓库 sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning 的 create_input_files.py 中词频统计与 word_map 生成逻辑），
并适配当前 Transformer 模型接口。

COPY Attribution:
- Source Repo: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning (MIT License)
- License: MIT (见原仓库 LICENSE)。本文件保留核心词频统计与特殊符号处理思路，代码已做最小化改写以嵌入当前项目结构。

差异说明:
- 原实现将图像与 captions 预处理为 HDF5+JSON。本处直接从 COCO annotations JSON 中收集 captions。
- 保留 <start>/<end>/<pad>/<unk> 四类特殊 token。
- 添加 encode/decode 接口与与 BertTokenizerWrapper 对齐的 API 以便可替换。
- 支持最小词频 min_freq、最大长度 max_len 过滤。

使用方式:
from src.datasets.word_vocab import build_word_vocab
vocab = build_word_vocab(train_ann_file, min_freq=5, max_len=50)
ids = vocab.encode("a man riding a horse")
text = vocab.decode(ids)

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter

@dataclass
class WordVocabConfig:
    min_freq: int = 5
    max_len: int = 50  # 过滤过长 caption（不含 <start>/<end>）
    lowercase: bool = True

class WordVocab:
    def __init__(self, word_map: Dict[str, int], inv_map: Dict[int, str]):
        self.word_map = word_map
        self.inv_map = inv_map
        # special ids
        self.pad_token = '<pad>'
        self.pad_token_id = self.word_map[self.pad_token]
        self.start_token = '<start>'
        self.start_token_id = self.word_map[self.start_token]
        self.end_token = '<end>'
        self.end_token_id = self.word_map[self.end_token]
        self.unk_token = '<unk>'
        self.unk_token_id = self.word_map[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.word_map)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if add_special_tokens:
            ids = [self.start_token_id]
        else:
            ids = []
        toks = self._tokenize(text)
        for t in toks:
            ids.append(self.word_map.get(t, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.end_token_id)
        return ids

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        for i in ids:
            if skip_special_tokens and i in {self.start_token_id, self.end_token_id, self.pad_token_id}:
                continue
            words.append(self.inv_map.get(i, self.unk_token))
        return ' '.join(words)

    def pad_sequences(self, sequences: List[List[int]], max_len: Optional[int] = None) -> List[List[int]]:
        if max_len is None:
            max_len = max(len(s) for s in sequences) if sequences else 0
        padded = []
        for s in sequences:
            if len(s) < max_len:
                padded.append(s + [self.pad_token_id] * (max_len - len(s)))
            else:
                padded.append(s[:max_len])
        return padded

    def _tokenize(self, text: str) -> List[str]:
        # 朴素按空格+基本字符过滤（Show, Attend and Tell 教程中类似做法）
        import re
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        toks = text.split()
        return toks

    @property
    def lowercase(self) -> bool:
        # 根据 '<unk>' 是否全小写推断；或直接返回 True
        return True


def build_word_vocab(ann_json_path: str, cfg: WordVocabConfig | Dict) -> WordVocab:
    if isinstance(cfg, dict):
        cfg = WordVocabConfig(**cfg)
    with open(ann_json_path, 'r') as f:
        data = json.load(f)
    annotations = data.get('annotations', [])
    counter = Counter()
    for ann in annotations:
        caption = ann.get('caption', '')
        toks = _basic_tokenize(caption, lowercase=cfg.lowercase)
        if cfg.max_len and len(toks) > cfg.max_len:
            continue
        counter.update(toks)
    # 构建词表（>= min_freq）
    words = [w for w, c in counter.items() if c >= cfg.min_freq]
    # 特殊 token 排在最前
    word_map: Dict[str, int] = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<unk>': 3,
    }
    for w in sorted(words):
        if w not in word_map:
            word_map[w] = len(word_map)
    inv_map = {i: w for w, i in word_map.items()}
    return WordVocab(word_map, inv_map)


def _basic_tokenize(text: str, lowercase: bool = True) -> List[str]:
    import re
    text = text.strip()
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    toks = text.split()
    return toks

__all__ = ["WordVocab", "WordVocabConfig", "build_word_vocab"]
