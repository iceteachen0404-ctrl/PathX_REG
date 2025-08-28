import torch
from safetensors.torch import save_model
from transformers import AutoModel


def main():
    m = AutoModel.from_pretrained('.', trust_remote_code=True)
    w = torch.load('pytorch_model.bin', map_location=torch.device('cpu'))
    m.load_state_dict(w['state_dict'])
    save_model(m, 'model.safetensors', metadata={'format': 'pt'})


if __name__ == '__main__':
    main()
