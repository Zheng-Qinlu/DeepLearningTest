"""
COCO 图像+字幕 Dataset / DataLoader 封装。

设计思路（参考/Copy 说明）：
- 数据划分与按注释 id 级别的样本构建，参考 aimagelab/meshed-memory-transformer 中 data/dataset.py: COCO 类
  （我们不复制其完整容器/Example/Field 抽象，只借鉴其将每条 caption 视作一个样本的做法）。

依赖：
- pycocotools
- Pillow
- torchvision.transforms
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pycocotools.coco import COCO

from src.datasets.vocab import BertTokenizerWrapper, TokenizerConfig


@dataclass
class CocoPaths:
	root: str
	train_dir: str
	val_dir: str
	ann_dir: str
	train_ann: str
	val_ann: str


def build_default_transforms(resize=(224, 224), imagenet_normalize=True):
	tfm = [T.Resize(resize), T.ToTensor()]
	if imagenet_normalize:
		tfm.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	return T.Compose(tfm)


class CocoCaptionDataset(Dataset):
	"""
	将 COCO 的每条 caption 注释作为一个样本（与 Meshed-Memory 的 get_samples 思路一致）。
	返回：image_tensor, input_ids
	"""

	def __init__(
		self,
		image_root: str,
		ann_file: str,
		tokenizer: BertTokenizerWrapper,
		transform=None,
		ann_ids: Optional[List[int]] = None,
		max_samples: Optional[int] = None,
	):
		super().__init__()
		self.image_root = image_root
		self.coco = COCO(ann_file)
		self.tokenizer = tokenizer
		self.transform = transform if transform is not None else build_default_transforms()

		if ann_ids is None:
			ann_ids = list(self.coco.anns.keys())
		# 固定顺序以便可复现切分
		ann_ids = list(ann_ids)
		if max_samples is not None:
			ann_ids = ann_ids[: int(max_samples)]
		self.ann_ids = ann_ids

	def __len__(self) -> int:
		return len(self.ann_ids)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		ann_id = self.ann_ids[idx]
		ann = self.coco.anns[ann_id]
		caption: str = ann["caption"]
		img_id = ann["image_id"]
		img_info = self.coco.loadImgs(img_id)[0]
		filename = img_info["file_name"]
		image_path = os.path.join(self.image_root, filename)

		image = Image.open(image_path).convert("RGB")
		image = self.transform(image)

		input_ids = self.tokenizer.encode(caption, add_special_tokens=True)

		return {
			"image": image,  # Tensor 3xHxW
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"ann_id": ann_id,
			"image_id": img_id,
		}

	@staticmethod
	def collate_fn(pad_token_id: int):
		def _collate(batch: List[Dict[str, Any]]):
			images = torch.stack([b["image"] for b in batch], dim=0)
			ids_list = [b["input_ids"].tolist() for b in batch]
			max_len = max(len(s) for s in ids_list)
			padded = []
			attn_mask = []
			for s in ids_list:
				pad_len = max_len - len(s)
				padded.append(s + [pad_token_id] * pad_len)
				attn_mask.append([1] * len(s) + [0] * pad_len)
			input_ids = torch.tensor(padded, dtype=torch.long)
			attention_mask = torch.tensor(attn_mask, dtype=torch.long)
			ann_ids = torch.tensor([b["ann_id"] for b in batch], dtype=torch.long)
			image_ids = torch.tensor([b["image_id"] for b in batch], dtype=torch.long)
			return {
				"images": images,
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"ann_id": ann_ids,
				"image_id": image_ids,
			}

		return _collate


def resolve_coco_paths(cfg: Dict[str, Any]) -> CocoPaths:
	root = cfg["data"]["root"]
	train_dir = os.path.join(root, cfg["data"]["images"]["train_dir"]) if isinstance(cfg["data"]["images"]["train_dir"], str) else cfg["data"]["images"]["train_dir"]
	val_dir = os.path.join(root, cfg["data"]["images"]["val_dir"]) if isinstance(cfg["data"]["images"]["val_dir"], str) else cfg["data"]["images"]["val_dir"]
	ann_dir = os.path.join(root, cfg["data"]["annotations"]["dir"]) if isinstance(cfg["data"]["annotations"]["dir"], str) else cfg["data"]["annotations"]["dir"]
	train_ann = os.path.join(ann_dir, cfg["data"]["annotations"]["train_file"]) if isinstance(cfg["data"]["annotations"]["train_file"], str) else cfg["data"]["annotations"]["train_file"]
	val_ann = os.path.join(ann_dir, cfg["data"]["annotations"]["val_file"]) if isinstance(cfg["data"]["annotations"]["val_file"], str) else cfg["data"]["annotations"]["val_file"]
	return CocoPaths(root, train_dir, val_dir, ann_dir, train_ann, val_ann)


def build_splits(
	cfg: Dict[str, Any],
	seed: int = 42,
	tokenizer: Optional[BertTokenizerWrapper] = None,
	transform=None,
) -> Tuple[CocoCaptionDataset, CocoCaptionDataset, CocoCaptionDataset]:
	"""
	构建 train/val/test 三个 Dataset。
	- 训练集：train2014 的所有注释
	- 验证与测试：来自 val2014，将其注释 id 随机打乱后，按 cfg 的比例划分（思路参考 Meshed-Memory 的 COCO.splits）
	"""
	paths = resolve_coco_paths(cfg)
	if tokenizer is None:
		tokenizer = BertTokenizerWrapper(cfg.get("tokenizer", {}))

	if transform is None:
		resize = tuple(cfg.get("transforms", {}).get("resize", [224, 224]))
		imagenet_norm = bool(cfg.get("transforms", {}).get("imagenet_normalize", True))
		transform = build_default_transforms(resize=resize, imagenet_normalize=imagenet_norm)

	# Train: 全部 train 注释
	coco_train = COCO(paths.train_ann)
	train_ann_ids = list(coco_train.anns.keys())
	max_train = cfg["data"].get("max_train_samples")
	train_dataset = CocoCaptionDataset(paths.train_dir, paths.train_ann, tokenizer, transform, ann_ids=train_ann_ids,
									   max_samples=max_train)

	# Val/Test: 来自 val 注释随机划分
	coco_val = COCO(paths.val_ann)
	val_ann_ids_all = list(coco_val.anns.keys())
	random.Random(seed).shuffle(val_ann_ids_all)
	val_ratio = float(cfg["data"].get("val_split", 0.8))
	split_idx = int(len(val_ann_ids_all) * val_ratio)
	val_ann_ids = val_ann_ids_all[:split_idx]
	test_ann_ids = val_ann_ids_all[split_idx:]

	max_val = cfg["data"].get("max_val_samples")
	max_test = cfg["data"].get("max_test_samples")

	val_dataset = CocoCaptionDataset(paths.val_dir, paths.val_ann, tokenizer, transform, ann_ids=val_ann_ids,
									 max_samples=max_val)
	test_dataset = CocoCaptionDataset(paths.val_dir, paths.val_ann, tokenizer, transform, ann_ids=test_ann_ids,
									  max_samples=max_test)

	return train_dataset, val_dataset, test_dataset

