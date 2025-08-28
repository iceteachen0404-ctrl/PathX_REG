import torch
from transformers import AutoModel

# Load Prism model.
model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
model = model.to('cuda')


# Load Virchow tile embeddings.
# See https://huggingface.co/paige-ai/Virchow on how to generate them
# given a whole slide image.
embedding_data = torch.load('E:/PyDL/CoCa-pytorch-main/prism/tcga/TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6.pth')
tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to('cuda')
print(tile_embeddings.shape)  # (batch_size, tile_seq_len, tile_embed_dim)
# > torch.Size([1, 12137, 2560])


# Compute slide embedding and latents. Only Perceiver is evaluated.
# We highly recommend running the model on a GPU in mixed precision (`fp16`)
# using `torch.autocast`.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    reprs = model.slide_representations(tile_embeddings)
print(reprs['image_embedding'].shape)
# > torch.Size([1, 1280])
print(reprs['image_latents'].shape)
# > torch.Size([1, 512, 1280])
print(reprs['image_embedding'][0, :8])
# > tensor([ 0.0620, -0.2983,  0.5849, -0.7383,  0.1242, -0.7440, -0.4719, -0.2920],
#          device='cuda:0')


# Do zero-shot prediction using the slide embedding.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    scores = model.zero_shot(
        reprs['image_embedding'],
        neg_prompts=['lobular carcinoma, invasive'],
        pos_prompts=['ductal carcinoma, invasive'],
    )
print(scores)
# > tensor([[0.0013, 0.9987]], device='cuda:0')


# Generate report using latent features.
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    genned_ids = model.generate(
        key_value_states=reprs['image_latents'],
        do_sample=False,
        num_beams=5,
        num_beam_groups=1,
    )
    genned_caption = model.untokenize(genned_ids)
print(genned_caption)
# > ['</s>Diagnosis: Moderately differentiated invasive ductal carcinoma '
# >  'with micropapillary features in breast tissue. </s>']


# Basic forward pass used in training.
# Computes slide embedding, text embedding, image latents (see Perceiver),
# next token logits, and similarity between slide and text embeddings
# used in contrastive alignment.
caption = model.tokenize(['some caption']).to('cuda')
with torch.autocast('cuda', torch.float16), torch.inference_mode():
    output = model(input_ids=caption, tile_embeddings=tile_embeddings)
print(output.keys())
# > dict_keys(['logits', 'text_embedding', 'image_embedding',
# >            'image_latents', 'sim'])