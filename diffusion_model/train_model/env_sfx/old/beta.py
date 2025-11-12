import torch
import math
import matplotlib.pyplot as plt

DEFAULTS = {
    'data_audio_dir': '../../data/MACS/audio',
    'data_annotation': '../../data/MACS/annotations/MACS.yaml',
    'n_mels': 80,
    'sample_rate': 16000,
    'duration': 10,
    'batch_size': 8,
    'epochs': 20,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'T': 1000,
    'base_ch': 256,
    't_dim': 256, #512
    'c_dim': 256, #512
    'save_dir': '../../output/sfx/weights',
    'sample_dir': '../../output/sfx/samples',
    'modelgen_dir': '../../output/sfx/test',
    'vocoder': 'griffinlim',
    'hifigan_ckpt': '',
    'grad_accum': 1,
    'mixed_precision': False,
    'use_global_norm': True,
    'stats_file': 'mel_stats.pt',
}

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-6, 0.999)

def analyze_diffusion_process(args):
    """分析扩散过程"""
    device = torch.device(args.device)
    
    # 测试不同的beta调度
    linear_betas = torch.linspace(1e-4, 0.02, args.T)
    cosine_betas = cosine_beta_schedule(args.T)
    
    # 计算累积alpha
    linear_alphas_cumprod = torch.cumprod(1 - linear_betas, dim=0)
    cosine_alphas_cumprod = torch.cumprod(1 - cosine_betas, dim=0)
    
    # 绘制调度比较
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(linear_betas.numpy(), label='Linear')
    plt.plot(cosine_betas.numpy(), label='Cosine')
    plt.title('Beta Schedules')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(linear_alphas_cumprod.numpy(), label='Linear')
    plt.plot(cosine_alphas_cumprod.numpy(), label='Cosine')
    plt.title('Alpha Cumulative Product')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('schedule_comparison.png')
    plt.close()
    
    print("Analysis saved to schedule_comparison.png")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','sample'], default='train')

    # paths and data
    parser.add_argument('--data_audio_dir', type=str, default=DEFAULTS['data_audio_dir'])
    parser.add_argument('--data_annotation', type=str, default=DEFAULTS['data_annotation'])
    parser.add_argument('--save_dir', type=str, default=DEFAULTS['save_dir'])
    parser.add_argument('--sample_dir', type=str, default=DEFAULTS['sample_dir'])
    parser.add_argument('--modelgen_dir', type=str, default=DEFAULTS['modelgen_dir'])

    # model / training
    parser.add_argument('--n_mels', type=int, default=DEFAULTS['n_mels'])
    parser.add_argument('--sample_rate', type=int, default=DEFAULTS['sample_rate'])
    parser.add_argument('--duration', type=int, default=DEFAULTS['duration'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    parser.add_argument('--T', type=int, default=DEFAULTS['T'])
    parser.add_argument('--base_ch', type=int, default=DEFAULTS['base_ch'])
    parser.add_argument('--t_dim', type=int, default=DEFAULTS['t_dim'])
    parser.add_argument('--c_dim', type=int, default=DEFAULTS['c_dim'])
    parser.add_argument('--vocoder', type=str, default=DEFAULTS['vocoder'])
    parser.add_argument('--hifigan_ckpt', type=str, default=DEFAULTS['hifigan_ckpt'])
    parser.add_argument('--grad_accum', type=int, default=DEFAULTS['grad_accum'])
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--use_global_norm', action='store_false')
    parser.add_argument('--stats_file', type=str, default=DEFAULTS['stats_file'])

    # sampling
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--text', type=str, default='a car is moving when it raining')
    parser.add_argument('--mel_len', type=int, default=None)
    args = parser.parse_args()

    # set flags
    args.mixed_precision = args.mixed_precision or DEFAULTS['mixed_precision']

    analyze_diffusion_process(args)