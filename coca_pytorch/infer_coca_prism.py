#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== 在导入 torch / transformers 之前设置环境变量 =====
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "6")  # 可按需改
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from typing import Optional, List, Dict

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 你的模型
from coca_pytorch_prism import CoCaPrism

# BioGPT tokenizer（与你训练时一致）
from transformers import BioGptTokenizer


# ---------- 数据集：仅用于推理（不需要 labels） ----------
class InferenceWSIDataset(Dataset):
    """
    只读取 .h5 embeddings 的推理数据集。
    假设每个 h5 里有键 'features'，形状 [num_tiles, tile_dim]。
    """
    def __init__(self, embeddings_dir: str, image_dim: int = 2560):
        self.emb_dir = Path(embeddings_dir)
        self.image_dim = image_dim

        # 收集 .h5 文件
        self.files = sorted([p for p in self.emb_dir.glob("*.h5")])
        if len(self.files) == 0:
            raise FileNotFoundError(f"未在 {self.emb_dir} 下找到 .h5 文件。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5_path = self.files[idx]
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]  # 期望是 [L, D]

        # 若数据是 [1, L, D]，去掉 batch 维；若是 [D] 或 [L] 则报错
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        assert feats.ndim == 2 and feats.shape[1] == self.image_dim, \
            f"{h5_path.name}: expect [L, {self.image_dim}], got {feats.shape}"

        emb = torch.from_numpy(feats).float()  # [L, D]
        # 单样本的 1D mask（True=有效）
        mask = torch.ones(emb.shape[0], dtype=torch.bool)  # [L]

        return {
            "image_embeddings": emb,        # [L, D]
            "image_tile_mask": mask,        # [L]
            "filename": h5_path.name
        }


# ---------- collate：pad 到同长并生成 batch ----------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    embs = [b["image_embeddings"] for b in batch]   # 每个 [L_i, D]
    masks = [b["image_tile_mask"] for b in batch]   # 每个 [L_i]
    filenames = [b["filename"] for b in batch]

    # [B, T, D]；pad 值 0.0
    image_embeddings = pad_sequence(embs, batch_first=True, padding_value=0.0)
    # [B, T]；pad 位置 False
    image_tile_mask = pad_sequence(masks, batch_first=True, padding_value=False)

    return {
        "image_embeddings": image_embeddings,   # [B, T, D]
        "image_tile_mask": image_tile_mask,     # [B, T], bool
        "filenames": filenames
    }


# ---------- 工具：解包 DP/DDP ----------
def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m


# ---------- 加载模型 ----------
def load_coca_prism_from_ckpt(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = CoCaPrism(
        prism_model_name=ckpt.get("prism_model_name", "./prism"),
        caption_loss_weight=ckpt.get("caption_loss_weight", 1.0),
        contrastive_loss_weight=ckpt.get("contrastive_loss_weight", 1.0),
        frozen_img=False,
        frozen_text_weights=True,
        frozen_text_embeddings=False
    )

    state = ckpt["model_state_dict"]
    # 去掉 DataParallel 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[Warn] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device).eval()
    return model


# ---------- 推理主函数 ----------
@torch.no_grad()
def run_inference(
    embeddings_dir: str,
    ckpt_path: str = "./checkpoints/checkpoints_soup_base_10_20_30_43/model_soup_by_regscore.pth",
    batch_size: int = 1,
    max_length: int = 50,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    save_path: Optional[str] = "./inference_outputs_final/checkpoints_soup_base_10_20_30_43.jsonl",
    num_workers: int = 2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # tokenizer（与你训练一致的 BioGPT）
    commit_hash = "eb0d815e95434dc9e3b78f464e52b899bee7d923"
    tokenizer = BioGptTokenizer.from_pretrained(
        "/data/case_level/open_source_model/model/biogpt",
        revision=commit_hash
    )

    # 数据
    ds = InferenceWSIDataset(embeddings_dir=embeddings_dir, image_dim=2560)
    logger.info(f'数据加载完成')
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=1,
        persistent_workers=True
    )
    logger.info(f'共找到测试数据：{len(ds)}')

    # 模型
    model = load_coca_prism_from_ckpt(ckpt_path, device)
    logger.info(f'模型加载成功')
    core = unwrap_model(model)

    results = []
    Path(save_path).parent.mkdir(parents=True, exist_ok=True) if save_path else None

    for batch in tqdm(dl, desc="Inference"):
        image_embeddings = batch["image_embeddings"].to(device)     # [B, T, D]
        tile_mask = batch["image_tile_mask"].to(device)             # [B, T], bool
        filenames = batch["filenames"]

        # autocast 推理（节省显存 / 更快）
        with autocast(device_type="cuda", dtype=torch.float16):
            generated_tokens = core.generate(
                image_tokens=image_embeddings,
                tile_mask=tile_mask,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.eos_token_id,   # 你的工程里 BioGPT 用 eos 作为 bos
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for fn, gen in zip(filenames, generated_texts):
            item = {"filename": fn, "generated": gen}
            results.append(item)
            if save_path:
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] 共生成 {len(results)} 条结果。")
    if save_path:
        print(f"[SAVED] 已保存到 {save_path}")
    return results


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    model_name = 'checkpoints0819_multibatch_train_seed60'
    ap = argparse.ArgumentParser(description="CoCaPrism 推理：只生成文本")
    ap.add_argument("--embeddings_dir", type=str, default='/data/slide_files/nas/vol2/Public_Data/reg2025/REG_test2_revised/trident_prcessed/20x_256px_0px_overlap/features_virchow/',
                    help="包含 .h5 特征的目录（键为 features）")
    ap.add_argument("--ckpt", type=str, default=f"/home/cjt/project_script/coca_pytorch/checkpoints/{model_name}/best_model_reg.pth",
                    help="训练好的 checkpoint 路径")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=50)
    ap.add_argument("--do_sample", action="store_true", help="是否采样生成（默认贪心）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--save_path", type=str, default=f"./inference_outputs_final/{model_name}.jsonl")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    run_inference(
        embeddings_dir=args.embeddings_dir,
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        save_path=args.save_path,
        num_workers=args.num_workers
    )
