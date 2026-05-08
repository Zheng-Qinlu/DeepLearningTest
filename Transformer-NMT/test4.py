from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
	import matplotlib
	matplotlib.use("Agg")  # 服务器/无显示环境
	import matplotlib.pyplot as plt
except Exception:
	plt = None


# ===============
# 基础工具
# ===============
def set_seed(seed: int = 2024):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True


def get_device(force_cpu: bool = False) -> torch.device:
	if not force_cpu and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


# ===============
# 数据路径与装载
# ===============
NIUTRANS_BASE = "https://raw.githubusercontent.com/NiuTrans/NiuTrans.SMT/master/sample-data"

FILES = {
	"train_zh": "TM-training-set/chinese.txt",
	"train_en": "TM-training-set/english.txt",
	"dev": "Dev-set/Niu.dev.txt",  # 有的版本用 “中文 ||| 英文” 一行表示
	"test": "Test-set/Niu.test.txt",
	"ref": "Reference-for-evaluation/Niu.test.reference",
}


def try_download(url: str, save_path: Path, timeout: int = 20) -> bool:
	try:
		import urllib.request
		save_path.parent.mkdir(parents=True, exist_ok=True)
		with urllib.request.urlopen(url, timeout=timeout) as r, open(save_path, "wb") as f:
			f.write(r.read())
		return True
	except Exception:
		return False


def prepare_data(root: Path) -> Dict[str, Path]:
	"""返回各数据文件路径；若不存在则尝试下载（需要网络）。"""
	paths: Dict[str, Path] = {}
	root = root.expanduser().resolve()
	for k, rel in FILES.items():
		p = root / rel
		if not p.exists():
			# 尝试下载
			url = f"{NIUTRANS_BASE}/{rel}"
			ok = try_download(url, p)
			if not ok:
				print(f"[WARN] 无法下载 {url}，请确认本地是否已有：{p}")
		if not p.exists():
			print(f"[WARN] 缺失文件：{p}")
		paths[k] = p
	return paths


# ===============
# 分词、词表
# ===============
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)


def tokenize(line: str) -> List[str]:
	# 数据已分词：中英均以空格分隔
	return line.strip().split()


def build_vocab(lines: Iterable[List[str]], max_size: int = 40000, min_freq: int = 1) -> Dict[str, int]:
	from collections import Counter

	counter = Counter()
	for toks in lines:
		counter.update(toks)
	items = [(w, c) for w, c in counter.items() if c >= min_freq]
	items.sort(key=lambda x: (-x[1], x[0]))
	itos = SPECIAL_TOKENS[:]
	for w, _ in items[: max(0, max_size - len(itos))]:
		itos.append(w)
	stoi = {w: i for i, w in enumerate(itos)}
	return stoi


def numericalize(tokens: List[str], vocab: Dict[str, int], add_sos: bool, add_eos: bool) -> List[int]:
	ids = [vocab.get(t, UNK) for t in tokens]
	if add_sos:
		ids = [SOS] + ids
	if add_eos:
		ids = ids + [EOS]
	return ids


# ===============
# 数据集与批处理
# ===============
class NMTDataset(Dataset):
	def __init__(self, src_lines: List[List[str]], trg_lines: Optional[List[List[str]]]):
		self.src = src_lines
		self.trg = trg_lines

	def __len__(self):
		return len(self.src)

	def __getitem__(self, idx):
		item = {"src": self.src[idx]}
		if self.trg is not None:
			item["trg"] = self.trg[idx]
		return item


def make_collate_fn(src_vocab: Dict[str, int], trg_vocab: Dict[str, int],
					max_src_len: int = 128, max_trg_len: int = 128):
	pad_id_src = PAD
	pad_id_trg = PAD

	def collate(batch):
		src_tok = [b["src"][:max_src_len] for b in batch]
		trg_tok = [b.get("trg") for b in batch]
		if trg_tok[0] is not None:
			trg_tok = [b[:max_trg_len] for b in trg_tok]
		else:
			trg_tok = None

		src_ids = [numericalize(x, src_vocab, add_sos=False, add_eos=True) for x in src_tok]
		if trg_tok is not None:
			y_full = [numericalize(y, trg_vocab, add_sos=False, add_eos=True) for y in trg_tok]
			y_in = [[SOS] + yy[:-1] for yy in y_full]
			y_out = y_full
		else:
			y_in = y_out = None

		def pad_to_max(seqs: List[List[int]], pad_id: int):
			max_len = max(len(s) for s in seqs)
			padded, mask = [], []
			for s in seqs:
				t = s + [pad_id] * (max_len - len(s))
				padded.append(t)
				mask.append([1] * len(s) + [0] * (max_len - len(s)))
			return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

		src_pad, src_nonpad = pad_to_max(src_ids, pad_id_src)
		if y_in is not None:
			y_in_pad, y_in_nonpad = pad_to_max(y_in, pad_id_trg)
			y_out_pad, _ = pad_to_max(y_out, pad_id_trg)
		else:
			y_in_pad = y_in_nonpad = y_out_pad = None

		out = {
			"src_ids": src_pad,
			"src_key_padding_mask": ~src_nonpad,  # True 表示 pad
		}
		if y_in_pad is not None:
			out.update({
				"tgt_in_ids": y_in_pad,
				"tgt_out_ids": y_out_pad,
				"tgt_key_padding_mask": ~y_in_nonpad,
			})
		return out

	return collate


# ===============
# 位置编码与模型
# ===============
class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # [1, L, D]
		self.register_buffer('pe', pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.pe[:, :x.size(1), :]
		return self.dropout(x)


class TransformerNMT(nn.Module):
	def __init__(self,
				 src_vocab_size: int,
				 trg_vocab_size: int,
				 d_model: int = 512,
				 nhead: int = 8,
				 num_encoder_layers: int = 6,
				 num_decoder_layers: int = 6,
				 dim_feedforward: int = 2048,
				 dropout: float = 0.1,
				 tie_embeddings: bool = True,
				 max_len: int = 512):
		super().__init__()
		self.d_model = d_model
		self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD)
		self.trg_embed = nn.Embedding(trg_vocab_size, d_model, padding_idx=PAD)
		self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_len)

		self.transformer = nn.Transformer(
			d_model=d_model,
			nhead=nhead,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			norm_first=False,
		)
		self.generator = nn.Linear(d_model, trg_vocab_size)
		if tie_embeddings:
			self.generator.weight = self.trg_embed.weight
		self._reset_parameters()

	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor,
				src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
		src = self.pos_enc(self.src_embed(src_ids) * math.sqrt(self.d_model))
		tgt = self.pos_enc(self.trg_embed(tgt_in_ids) * math.sqrt(self.d_model))
		T = tgt_in_ids.size(1)
		causal_mask = torch.triu(torch.ones(T, T, device=tgt.device), diagonal=1).bool()
		out = self.transformer(
			src=src,
			tgt=tgt,
			src_key_padding_mask=src_key_padding_mask,
			tgt_key_padding_mask=tgt_key_padding_mask,
			memory_key_padding_mask=src_key_padding_mask,
			tgt_mask=causal_mask,
		)
		logits = self.generator(out)
		return logits

	@torch.no_grad()
	def greedy_decode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor,
					  max_len: int = 128) -> torch.Tensor:
		B = src_ids.size(0)
		device = src_ids.device
		src = self.pos_enc(self.src_embed(src_ids) * math.sqrt(self.d_model))
		memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
		ys = torch.full((B, 1), SOS, dtype=torch.long, device=device)
		finished = torch.zeros(B, dtype=torch.bool, device=device)
		for _ in range(max_len):
			tgt = self.pos_enc(self.trg_embed(ys) * math.sqrt(self.d_model))
			T = tgt.size(1)
			causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
			out = self.transformer.decoder(
				tgt=tgt,
				memory=memory,
				tgt_mask=causal_mask,
				tgt_key_padding_mask=torch.zeros(B, T, dtype=torch.bool, device=device),
				memory_key_padding_mask=src_key_padding_mask,
			)
			logits = self.generator(out[:, -1:, :])
			next_token = torch.argmax(logits.squeeze(1), dim=-1)
			ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
			finished |= (next_token == EOS)
			if torch.all(finished):
				break
		return ys[:, 1:]


# ===============
# Noam 学习率调度
# ===============
class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
		self.d_model = d_model
		self.warmup = warmup_steps
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		step = max(1, self._step_count)
		scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
		return [base_lr * scale for base_lr in self.base_lrs]


# ===============
# BLEU-4（Corpus）
# ===============
from collections import Counter


def _ngram_counts(tokens: List[str], n: int) -> Counter:
	return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])


def corpus_bleu_4(hyps: List[List[str]], refs: List[List[str]], smooth: bool = True) -> float:
	weights = [0.25, 0.25, 0.25, 0.25]
	clipped = [0, 0, 0, 0]
	total = [0, 0, 0, 0]
	hyp_len = 0
	ref_len = 0
	for h, r in zip(hyps, refs):
		hyp_len += len(h)
		ref_len += len(r)
		for n in range(1, 5):
			hc = _ngram_counts(h, n)
			rc = _ngram_counts(r, n)
			total[n-1] += max(sum(hc.values()), 1)
			overlap = {g: min(c, rc.get(g, 0)) for g, c in hc.items()}
			clipped[n-1] += sum(overlap.values())
	prec = []
	for i in range(4):
		if smooth:
			p = (clipped[i] + 1.0) / (total[i] + 1.0)
		else:
			p = clipped[i] / total[i] if total[i] > 0 else 0.0
		prec.append(p)
	bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len + 1e-16) / (hyp_len + 1e-16))
	bleu = bp * math.exp(sum(w * math.log(p + 1e-16) for w, p in zip(weights, prec)))
	return bleu * 100.0


# ===============
# 读取数据文件
# ===============
def load_parallel_train(train_zh_path: Path, train_en_path: Path,
						max_samples: Optional[int] = None) -> Tuple[List[List[str]], List[List[str]]]:
	zh_lines, en_lines = [], []
	with open(train_zh_path, "r", encoding="utf-8") as fz, open(train_en_path, "r", encoding="utf-8") as fe:
		for z, e in zip(fz, fe):
			zt, et = tokenize(z), tokenize(e)
			if zt and et:
				zh_lines.append(zt)
				en_lines.append(et)
			if max_samples is not None and len(zh_lines) >= max_samples:
				break
	return zh_lines, en_lines


def load_dev_pairs(dev_path: Path) -> Optional[Tuple[List[List[str]], List[List[str]]]]:
	if not dev_path.exists():
		return None
	zh, en = [], []
	with open(dev_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if "|||" in line:
				a, b = line.split("|||")
				zh.append(tokenize(a))
				en.append(tokenize(b))
			else:
				# 不符合期望格式则返回 None
				return None
	return zh, en


def load_test_and_ref(test_path: Path, ref_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
	"""
	读取测试集与参考翻译。

	注意：NiuTrans sample 的 Reference-for-evaluation/Niu.test.reference 文件是“中文/英文交替行”。
	若直接逐行读取，会将中文行也当作参考英文，从而导致测试 BLEU 异常偏低（接近 0）。

	这里做一个鲁棒处理：
	- 优先读取测试源句（中文）。
	- 读取 reference 后，若参考行数约为测试源的 2 倍，则认为为“中/英交替行”，仅保留偶数行（从第 2 行开始，每隔一行）。
	- 否则，按普通单语参考处理。
	"""
	test_zh: List[List[str]] = []
	refs_en: List[List[str]] = []

	# 读取测试中文源
	if test_path.exists():
		with open(test_path, "r", encoding="utf-8") as f:
			for line in f:
				t = tokenize(line)
				if t:
					test_zh.append(t)

	# 读取参考，考虑“中/英交替行”情况
	if ref_path.exists():
		with open(ref_path, "r", encoding="utf-8") as f:
			raw_ref_lines = [ln.strip() for ln in f if ln.strip()]

		# 如果参考行数约为测试源的两倍，认为是“中/英交替行”，仅保留偶数（英文）行
		if test_zh and len(raw_ref_lines) >= 2 * len(test_zh) - 5:
			eng_lines = raw_ref_lines[1::2]  # 从第2行开始每隔一行（英文）
		else:
			eng_lines = raw_ref_lines

		for line in eng_lines:
			t = tokenize(line)
			if t:
				refs_en.append(t)

	# 若存在长度差异，进行对齐（以较短者为准）
	if refs_en and test_zh:
		n = min(len(test_zh), len(refs_en))
		test_zh = test_zh[:n]
		refs_en = refs_en[:n]

	return test_zh, refs_en


# ===============
# 训练与评估
# ===============
def train_one_epoch(model: TransformerNMT, loader: DataLoader, optimizer, scheduler, device: torch.device,
					label_smoothing: float = 0.0) -> Tuple[float, float]:
	model.train()
	total_loss = 0.0
	total_tokens = 0
	correct_tokens = 0
	for batch in loader:
		src_ids = batch["src_ids"].to(device)
		tgt_in = batch["tgt_in_ids"].to(device)
		tgt_out = batch["tgt_out_ids"].to(device)
		src_pad = batch["src_key_padding_mask"].to(device)
		tgt_pad = batch["tgt_key_padding_mask"].to(device)

		logits = model(src_ids, tgt_in, src_pad, tgt_pad)
		V = logits.size(-1)
		loss = F.cross_entropy(
			logits.reshape(-1, V),
			tgt_out.reshape(-1),
			ignore_index=PAD,
			label_smoothing=label_smoothing if hasattr(F, "cross_entropy") else 0.0,
		)
		# 教师强制的 token 级准确率
		with torch.no_grad():
			pred = logits.argmax(-1)
			mask = tgt_out.ne(PAD)
			correct_tokens += (pred.eq(tgt_out) & mask).sum().item()
			total_tokens += mask.sum().item()
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		if scheduler is not None:
			scheduler.step()
		total_loss += loss.item()
	train_acc = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
	return total_loss / max(1, len(loader)), train_acc


@torch.no_grad()
def evaluate_bleu_and_accuracy(model: TransformerNMT, loader: DataLoader, device: torch.device,
								 trg_vocab_itow: List[str]) -> Tuple[float, float, float]:
	"""
	返回 (BLEU-4, token级准确率, 句子完全匹配准确率)，基于贪心译文与参考的对齐。
	"""
	model.eval()
	hyps, refs = [], []
	token_correct = 0
	token_total = 0
	sent_correct = 0
	sent_total = 0
	for batch in loader:
		src_ids = batch["src_ids"].to(device)
		src_pad = batch["src_key_padding_mask"].to(device)
		pred_ids = model.greedy_decode(src_ids, src_pad, max_len=128)
		batch_hyps: List[List[str]] = []
		for i in range(pred_ids.size(0)):
			ids = pred_ids[i].tolist()
			toks = []
			for t in ids:
				if t == EOS:
					break
				if t in (PAD, SOS):
					continue
				toks.append(trg_vocab_itow[t] if 0 <= t < len(trg_vocab_itow) else "<unk>")
			batch_hyps.append(toks)
		hyps.extend(batch_hyps)

		if "tgt_out_ids" in batch:
			tgt_out = batch["tgt_out_ids"]
			for i in range(tgt_out.size(0)):
				gold = []
				for t in tgt_out[i].tolist():
					if t in (PAD, SOS):
						continue
					if t == EOS:
						break
					gold.append(trg_vocab_itow[t])
				refs.append(gold)
				if i < len(batch_hyps):
					pred = batch_hyps[i]
					m = min(len(pred), len(gold))
					token_correct += sum(1 for k in range(m) if pred[k] == gold[k])
					token_total += len(gold)
					sent_correct += 1 if pred == gold else 0
					sent_total += 1
	bleu = corpus_bleu_4(hyps, refs) if refs else 0.0
	token_acc = (token_correct / token_total) if token_total > 0 else 0.0
	sent_acc = (sent_correct / sent_total) if sent_total > 0 else 0.0
	return bleu, token_acc, sent_acc


def save_curves_csv_png(history: List[Dict[str, float]], out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)
	csv_path = out_dir / "training_metrics.csv"
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "dev_bleu", "dev_token_acc", "dev_seq_acc"])
		writer.writeheader()
		for row in history:
			writer.writerow(row)
	if plt is None:
		print("[WARN] matplotlib 不可用，跳过训练曲线绘制。")
		return
	xs = [r["epoch"] for r in history]
	train_loss = [r.get("train_loss", 0.0) for r in history]
	dev_bleu = [r.get("dev_bleu", 0.0) for r in history]
	fig, ax1 = plt.subplots(figsize=(8, 4))
	ax1.plot(xs, train_loss, "-o", color="tab:blue", label="Train Loss")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Loss", color="tab:blue")
	ax1.tick_params(axis='y', labelcolor='tab:blue')
	ax2 = ax1.twinx()
	ax2.plot(xs, dev_bleu, "-s", color="tab:orange", label="Dev BLEU-4")
	ax2.set_ylabel("BLEU-4", color="tab:orange")
	ax2.tick_params(axis='y', labelcolor='tab:orange')
	fig.tight_layout()
	fig.savefig(out_dir / "training_curves.png", dpi=150)
	plt.close(fig)


# ===============
# 主流程
# ===============
def main():
	parser = argparse.ArgumentParser(description="Transformer NMT (Zh->En) on NiuTrans sample-data")
	default_data = Path(__file__).parent / "sample-submission-version"
	parser.add_argument("--data-dir", type=str, default=str(default_data))
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--d-model", type=int, default=512)
	parser.add_argument("--nhead", type=int, default=8)
	parser.add_argument("--num-enc-layers", type=int, default=4)
	parser.add_argument("--num-dec-layers", type=int, default=4)
	parser.add_argument("--ffn-dim", type=int, default=2048)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--max-vocab-src", type=int, default=40000)
	parser.add_argument("--max-vocab-trg", type=int, default=40000)
	parser.add_argument("--max-src-len", type=int, default=128)
	parser.add_argument("--max-trg-len", type=int, default=128)
	parser.add_argument("--warmup-steps", type=int, default=4000)
	parser.add_argument("--lr", type=float, default=2.0)  # Noam base lr
	parser.add_argument("--seed", type=int, default=2024)
	parser.add_argument("--force-cpu", action="store_true")
	parser.add_argument("--save-dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
	parser.add_argument("--eval-only", action="store_true", help="仅评估（需已有模型）")
	# 加载已保存模型参数
	parser.add_argument("--init-from-best", action="store_true", help="若存在最优模型，则在训练前加载其参数作为初始化")
	parser.add_argument("--no-ask", action="store_true", help="不进行交互式询问是否加载已有模型")
	# 早停与可视化
	parser.add_argument("--early-stop-patience", type=int, default=3, help="连续多少轮改进小于阈值则早停")
	parser.add_argument("--early-stop-delta", type=float, default=1.0, help="认为显著改进所需的最小 BLEU 提升")
	parser.add_argument("--show-samples", type=int, default=5, help="训练/评估结束后展示的翻译样例数量")
	args = parser.parse_args()
	# 限制最大学习轮数为 30
	if args.epochs > 30:
		print("[INFO] epochs 超过上限 30，已截断为 30")
		args.epochs = 30

	set_seed(args.seed)
	device = get_device(force_cpu=args.force_cpu)
	print(f"[INFO] Device: {device}")

	data_root = Path(args.data_dir)
	paths = prepare_data(data_root)

	train_zh, train_en = load_parallel_train(paths["train_zh"], paths["train_en"], max_samples=None)
	dev_pair = load_dev_pairs(paths["dev"])  # 可能为 None
	test_zh, test_ref_en = load_test_and_ref(paths["test"], paths["ref"])  # 参考可能为空

	if dev_pair is None:
		cut = min(2000, int(0.02 * len(train_zh)))
		dev_zh, dev_en = train_zh[:cut], train_en[:cut]
		tr_zh, tr_en = train_zh[cut:], train_en[cut:]
	else:
		dev_zh, dev_en = dev_pair
		tr_zh, tr_en = train_zh, train_en

	print(f"[INFO] Train pairs: {len(tr_zh)}, Dev pairs: {len(dev_zh)}, Test zh: {len(test_zh)}")

	# 构建词表
	src_vocab = build_vocab(tr_zh, max_size=args.max_vocab_src, min_freq=1)
	trg_vocab = build_vocab(tr_en, max_size=args.max_vocab_trg, min_freq=1)
	itos_trg = [None] * len(trg_vocab)
	for w, i in trg_vocab.items():
		itos_trg[i] = w

	# 数据集 & DataLoader
	train_ds = NMTDataset(tr_zh, tr_en)
	dev_ds = NMTDataset(dev_zh, dev_en)
	collate = make_collate_fn(src_vocab, trg_vocab, args.max_src_len, args.max_trg_len)
	pin = (device.type == "cuda")
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
							  pin_memory=pin, collate_fn=collate, drop_last=True)
	dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
							pin_memory=pin, collate_fn=collate)

	# 模型
	model = TransformerNMT(
		src_vocab_size=len(src_vocab),
		trg_vocab_size=len(trg_vocab),
		d_model=args.d_model,
		nhead=args.nhead,
		num_encoder_layers=args.num_enc_layers,
		num_decoder_layers=args.num_dec_layers,
		dim_feedforward=args.ffn_dim,
		dropout=args.dropout,
		tie_embeddings=True,
		max_len=max(args.max_src_len, args.max_trg_len) + 10,
	).to(device)

	save_dir = Path(args.save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)
	best_path = save_dir / "nmt_best.pt"

	# 若存在 checkpoint，开头询问/或根据参数决定是否加载
	if best_path.exists() and not args.eval_only:
		if args.init_from_best:
			print(f"[INFO] 将从已保存最优模型初始化参数: {best_path}")
		elif not args.no_ask and sys.stdin.isatty():
			try:
				ans = input(f"[PROMPT] 发现已保存模型参数 {best_path.name}。是否加载其参数作为训练初始化? [y/N]，或输入 E 仅评估后退出: ").strip().lower()
			except EOFError:
				ans = ""
			if ans == "e":
				args.eval_only = True
				print("[INFO] 选择仅评估模式。")
			elif ans in ("y", "yes"): 
				args.init_from_best = True
				print("[INFO] 训练前将加载已保存参数作为初始化。")

	# 如需，用已保存参数初始化模型
	if best_path.exists() and args.init_from_best and not args.eval_only:
		state = torch.load(best_path, map_location=device)
		missing, unexpected = model.load_state_dict(state["model"], strict=False)
		print(f"[INFO] 已加载初始化参数。missing={len(missing)} unexpected={len(unexpected)}")

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
	scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

	history: List[Dict[str, float]] = []
	best_bleu = -1.0
	patience = 0

	if args.eval_only and best_path.exists():
		state = torch.load(best_path, map_location=device)
		model.load_state_dict(state["model"])
		print("[INFO] 已加载最优模型，开始评估 Dev 集…")
		dev_bleu, dev_token_acc, dev_seq_acc = evaluate_bleu_and_accuracy(model, dev_loader, device, itos_trg)
		print(f"[EVAL] Dev BLEU-4: {dev_bleu:.2f} TokAcc={dev_token_acc*100:.2f}% SeqAcc={dev_seq_acc*100:.2f}%")
	else:
		for epoch in range(1, args.epochs + 1):
			t0 = time.time()
			train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, label_smoothing=0.1)
			t1 = time.time()
			dev_bleu, dev_token_acc, dev_seq_acc = evaluate_bleu_and_accuracy(model, dev_loader, device, itos_trg)
			t2 = time.time()
			history.append({"epoch": epoch, "train_loss": float(train_loss), "train_acc": float(train_acc),
						"dev_bleu": float(dev_bleu), "dev_token_acc": float(dev_token_acc), "dev_seq_acc": float(dev_seq_acc)})
			improve = dev_bleu - (best_bleu if best_bleu >= 0 else 0.0)
			print(f"[EPOCH {epoch:02d}] Loss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% (time {t1-t0:.1f}s)  Dev BLEU-4={dev_bleu:.2f} TokAcc={dev_token_acc*100:.2f}% SeqAcc={dev_seq_acc*100:.2f}%  Δ={improve:.2f} (time {t2-t1:.1f}s)")

			# 保存更优模型（提升大小不做限制）
			if dev_bleu > best_bleu:
				best_bleu = dev_bleu
				torch.save({"model": model.state_dict(), "src_vocab": src_vocab, "trg_vocab": trg_vocab}, best_path)
				print(f"[INFO] 保存最优模型到 {best_path} (BLEU-4={best_bleu:.2f})")

			# 新早停逻辑：当 BLEU 已达到阈值(≥14)后，若连续若干轮提升 <0.3 则停止
			# 需求：BLEU-4 值大于 14 并且连续 3 轮提升小于 0.3 早停
			if dev_bleu >= 14.0 and improve < 0.3:
				patience += 1
				print(f"[EARLY-STOP] BLEU≥14 且提升<0.3 连续计数 {patience}/3")
			else:
				patience = 0
			if patience >= 3:
				print("[EARLY-STOP] BLEU≥14 后连续 3 轮提升 <0.3, 提前停止训练。")
				break
		if args.epochs > 0:
			save_curves_csv_png(history, out_dir=save_dir)

	# =============
	# 测试集评估
	# =============
	if best_path.exists():
		state = torch.load(best_path, map_location=device)
		model.load_state_dict(state["model"])  # 使用最优权重

	# 准备仅含源端的测试 DataLoader
	test_zh, test_ref_en = load_test_and_ref(paths["test"], paths["ref"])  # 重新确保最新
	test_ds = NMTDataset(test_zh, None)

	def collate_test(batch):
		src_tok = [b["src"][:args.max_src_len] for b in batch]
		src_ids = [numericalize(x, src_vocab, add_sos=False, add_eos=True) for x in src_tok]
		if not src_ids:
			src_ids = [[EOS]]
		max_len = max(len(s) for s in src_ids)
		padded = [s + [PAD] * (max_len - len(s)) for s in src_ids]
		mask = [[1] * len(s) + [0] * (max_len - len(s)) for s in src_ids]
		return {
			"src_ids": torch.tensor(padded, dtype=torch.long),
			"src_key_padding_mask": ~torch.tensor(mask, dtype=torch.bool),
		}

	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
							 pin_memory=(device.type == "cuda"), collate_fn=collate_test)

	test_bleu = None
	test_token_acc = None
	test_seq_acc = None
	if test_ref_en:
		model.eval()
		hyps: List[List[str]] = []
		with torch.no_grad():
			for batch in test_loader:
				src_ids = batch["src_ids"].to(device)
				src_pad = batch["src_key_padding_mask"].to(device)
				pred_ids = model.greedy_decode(src_ids, src_pad, max_len=args.max_trg_len)
				for i in range(pred_ids.size(0)):
					ids = pred_ids[i].tolist()
					toks = []
					for t in ids:
						if t == EOS:
							break
						if t in (PAD, SOS):
							continue
						toks.append(itos_trg[t] if 0 <= t < len(itos_trg) else "<unk>")
					hyps.append(toks)
		n = min(len(hyps), len(test_ref_en))
		test_bleu = corpus_bleu_4(hyps[:n], test_ref_en[:n])
		# 准确率
		tok_corr = 0
		tok_total = 0
		seq_corr = 0
		for i in range(n):
			pred = hyps[i]
			gold = test_ref_en[i]
			m = min(len(pred), len(gold))
			tok_corr += sum(1 for k in range(m) if pred[k] == gold[k])
			tok_total += len(gold)
			seq_corr += 1 if pred == gold else 0
		test_token_acc = (tok_corr / tok_total) if tok_total > 0 else 0.0
		test_seq_acc = (seq_corr / n) if n > 0 else 0.0
		print(f"[TEST] BLEU-4: {test_bleu:.2f}  TokAcc={test_token_acc*100:.2f}%  SeqAcc={test_seq_acc*100:.2f}%")
	else:
		print("[WARN] 缺少测试集参考翻译 Reference-for-evaluation/Niu.test.reference，跳过测试 BLEU。")

	# 汇总输出
	if test_bleu is not None:
		print(f"[SUMMARY] Test BLEU-4: {test_bleu:.2f}")

	# 翻译样例输出（从测试集，若为空则用开发集）
	def print_samples_from(src_list: List[List[str]], ref_list: Optional[List[List[str]]], n: int):
		model.eval()
		print("\n[SAMPLES] 翻译样例：")
		for i in range(min(n, len(src_list))):
			src_tok = src_list[i][:args.max_src_len]
			src_ids = torch.tensor([numericalize(src_tok, src_vocab, add_sos=False, add_eos=True)], dtype=torch.long, device=device)
			src_mask = torch.zeros(1, src_ids.size(1), dtype=torch.bool, device=device)
			pred_ids = model.greedy_decode(src_ids, src_mask, max_len=args.max_trg_len)[0].tolist()
			pred_tok = []
			for t in pred_ids:
				if t == EOS:
					break
				if t in (PAD, SOS):
					continue
				pred_tok.append(itos_trg[t] if 0 <= t < len(itos_trg) else "<unk>")
			ref_str = ("  |  REF: " + " ".join(ref_list[i])) if (ref_list and i < len(ref_list)) else ""
			print(f"SRC: {' '.join(src_tok)}\nPRED:{' '.join(pred_tok)}{ref_str}\n---")

	if len(test_zh) > 0:
		print_samples_from(test_zh, test_ref_en if test_ref_en else None, args.show_samples)
	elif len(dev_zh) > 0:
		print_samples_from(dev_zh, dev_en, args.show_samples)


if __name__ == "__main__":
	main()


