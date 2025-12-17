# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import time

wandb_log = True
# wandb_project = 'owt'
# wandb_run_name='gpt2-124M'
wandb_project = "no-ffn-gpt2"
wandb_run_name = "no-ffn-gpt2_shakespeare_" + str(time.time())
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# batch_size = 12
# block_size = 1024
# gradient_accumulation_steps = 5 * 8


gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher

out_dir = "out-shakespeare"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often
