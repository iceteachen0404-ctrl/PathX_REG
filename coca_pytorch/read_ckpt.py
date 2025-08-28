import torch
path = '/home/cjt/project_script/coca_pytorch/checkpoints/checkpoints0730_train_seed10/best_model_reg.pth'
ckpt = torch.load(path, map_location='cpu')
print(ckpt['reg_score'])