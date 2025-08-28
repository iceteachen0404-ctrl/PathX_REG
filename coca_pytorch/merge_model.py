import torch
import os

model_paths = [
    "./checkpoints/checkpoints0730_train_seed10/best_model_reg.pth",
    # "./checkpoints/checkpoints0730_train_seed20/best_model_reg.pth",
    "./checkpoints/checkpoints0730_train_seed30/best_model_reg.pth"
    # "./checkpoints/checkpoints0730_train_seed43/best_model_reg.pth"
    # "./checkpoints/checkpoints0730/best_model_reg.pth"
]

# 读取所有 state_dict 和对应的 reg_score
state_dicts = []
scores = []

for path in model_paths:
    ckpt = torch.load(path, map_location='cpu')
    state_dicts.append(ckpt['model_state_dict'])
    scores.append(ckpt['reg_score'])
    print(f"{path} REG Score: {ckpt['reg_score']}")

# 转成 tensor
scores_tensor = torch.tensor(scores, dtype=torch.float32)
weights = scores_tensor / scores_tensor.sum()  # 归一化

# 初始化 weighted state dict
keys = state_dicts[0].keys()
soup_state_dict = {
    key: torch.zeros_like(state_dicts[0][key]) for key in keys
}

# 加权平均
for state_dict, weight in zip(state_dicts, weights):
    for key in keys:
        soup_state_dict[key] += state_dict[key] * weight

# 保存融合模型
base_ckpt = torch.load(model_paths[0], map_location='cpu')  # 用第一个模型结构当模板
base_ckpt['model_state_dict'] = soup_state_dict
from pathlib import Path
save_base_dir = Path('./checkpoints/checkpoints_soup_10_30_43/')
save_base_dir.mkdir(parents=True, exist_ok=True)
base_ckpt['reg_score'] = float(scores_tensor.mean())  # 可以记录平均得分或其他指标
torch.save(base_ckpt, os.path.join(save_base_dir, 'model_soup_by_regscore.pth'))
print("✅ 加权融合模型已保存为 model_soup_weighted_by_regscore.pth")