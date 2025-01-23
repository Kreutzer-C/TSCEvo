import torch
import torch.nn.functional as F


def get_entropy_conf(p):
    u = -1.0 * p * torch.log(p + 1e-6)
    conf = 1 - u
    return conf


def do_collaboration(p_tar, p_clip, cp_mb, alpha=0.5):
    if alpha == 0:
        # Entropy-based prediction mixture
        conf_tar = get_entropy_conf(p_tar)
        conf_clip = get_entropy_conf(p_clip)
        p_mix = (conf_tar * p_tar + conf_clip * p_clip) / (conf_tar + conf_clip)

    else:
        # Class prototype Attention
        p_tar_norm = p_tar / p_tar.norm(dim=1, keepdim=True)
        cp_mb_norm = cp_mb / cp_mb.norm(dim=1, keepdim=True)
        sim = torch.matmul(p_tar_norm, cp_mb_norm.T)  # N*C
        sim = F.softmax(sim, dim=1)
        weight = cp_mb.permute(1, 0)
        p_tar_new = F.linear(sim, weight)

        # Updata cp memory
        y_tar = torch.argmax(p_tar, dim=1)
        y_clip = torch.argmax(p_clip, dim=1)
        for i in range(p_tar.size(1)):
            indices = (y_tar == i) & (y_clip == i)

            if indices.any():
                selected_p_tar = p_tar[indices]
                cp_mb[i] = (selected_p_tar.sum(dim=0) + cp_mb[i]) / (1 + selected_p_tar.size(0))

        # Entropy-based prediction mixture
        conf_tar = get_entropy_conf(p_tar_new)
        conf_clip = get_entropy_conf(p_clip)
        p_mix = (conf_tar * p_tar_new + conf_clip * p_clip) / (conf_tar + conf_clip)

    return p_mix, cp_mb
