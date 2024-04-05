import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np 

from utils import init_weights, get_padding
from models import LRELU_SLOPE, Encoder, ResBlock1, ResBlock2, SourceModuleHnNSF


class Generator1(torch.nn.Module):
    def __init__(self, h, wavlm, F0_model):
        super().__init__()
        self.h = h

        self.wavlm = wavlm
        self.F0_model = F0_model

        gin_channels = 256
        inter_channels = hidden_channels = h.upsample_initial_channel - gin_channels

        self.embed_spk = nn.Embedding(108, gin_channels)
        self.enc = Encoder(768, inter_channels, hidden_channels, 5, 1, 4) 
        self.dec = Encoder(inter_channels, inter_channels, hidden_channels, 5, 1, 20, gin_channels=gin_channels) 

        self.m_source = SourceModuleHnNSF(
                    sampling_rate=h.sampling_rate,
                    upsample_scale=np.prod(h.upsample_rates) * h.gen_istft_hop_size,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h.upsample_rates) * h.gen_istft_hop_size)

    def get_x(self, wav, spk_emb, spk_id):
        x = self.wavlm(wav).last_hidden_state
        x = x.transpose(1, 2) # (B, C, T)
        x = F.pad(x, (0, wav.size(-1) // 320 - x.size(2)), 'constant')

        g = self.embed_spk(spk_id).transpose(1, 2)
        g = g + spk_emb.unsqueeze(-1)

        x = self.enc(x)
        x = self.dec(x, g=g)
        g = g.repeat(1, 1, x.shape[-1])
        x = torch.cat([x, g], dim=1)

        return x

    def get_f0(self, mel, f0_mean_tgt):
        voiced_threshold = 10

        f0, _, _ = self.F0_model(mel.unsqueeze(1))
        voiced = f0 > voiced_threshold

        lf0 = torch.log(f0)
        lf0_ = lf0 * voiced.float()
        lf0_mean = lf0_.sum(1) / voiced.float().sum(1) 
        lf0_mean = lf0_mean.unsqueeze(1)
        lf0_adj = lf0 - lf0_mean + torch.log(f0_mean_tgt)
        f0_adj = torch.exp(lf0_adj)

        energy = mel.sum(1)
        unsilent = energy > -700
        unsilent = unsilent | voiced    # simple vad
        f0_adj = f0_adj * unsilent.float()

        return f0_adj

    def get_har(self, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)

        return har_source

    def forward(self, wav, mel, spk_emb, spk_id, f0_mean_tgt):
        x = self.get_x(wav, spk_emb, spk_id)
        f0 = self.get_f0(mel, f0_mean_tgt)
        har_source = self.get_har(f0)

        return x, har_source


class Generator2(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = np.prod(h.upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    h.gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(h, c_cur, 7, [1,3,5]))
            else:
                self.noise_convs.append(Conv1d(h.gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(resblock(h, c_cur, 11, [1,3,5]))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.post_n_fft = h.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
    
    def forward(self, x, har_spec, har_phase):
        har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source)
            
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
