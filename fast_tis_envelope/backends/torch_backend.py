"""Scaled PyTorch implementation of the maximum TIS envelope."""

import torch


def envelop_torch(e1, e2):
    if isinstance(e1, torch.Tensor) and isinstance(e2, torch.Tensor) and e1.device != e2.device:
        raise ValueError("e1 and e2 must be on the same device")
    device = e1.device if isinstance(e1, torch.Tensor) else e2.device if isinstance(e2, torch.Tensor) else None
    a, b = torch.as_tensor(e1, device=device), torch.as_tensor(e2, device=device)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape:
        raise ValueError("e1 and e2 must have identical shape (N, 3)")
    if a.is_complex() or b.is_complex():
        raise TypeError("e1 and e2 must be real")
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), torch.float32)
    a, b = a.to(dtype), b.to(dtype)
    sa, sb = a.abs().amax(dim=1), b.abs().amax(dim=1)
    x = a / torch.where(sa > 0, sa, 1)[:, None]
    y = b / torch.where(sb > 0, sb, 1)[:, None]
    flip = (x * y).sum(dim=1) < 0
    x = torch.where(flip[:, None], -x, x)
    a = torch.where(flip[:, None], -a, a)
    na, nb = torch.linalg.vector_norm(x, dim=1), torch.linalg.vector_norm(y, dim=1)
    equal = (a == b).all(dim=1)
    d = a * 0.5 - b * 0.5
    ds = d.abs().amax(dim=1)
    d = d / torch.where(ds > 0, ds, 1)[:, None]
    den = torch.linalg.vector_norm(d, dim=1)
    use_a = sa < sb
    cross = torch.linalg.cross(torch.where(use_a[:, None], x, y), d, dim=1)
    general = torch.where(use_a, sa, sb) * (2 * torch.linalg.vector_norm(cross, dim=1) / torch.where(den > 0, den, 1))
    return torch.where(equal, sa * na + sb * nb,
        torch.where((y * d).sum(dim=1) > 0, sb * (2 * nb),
            torch.where((x * d).sum(dim=1) < 0, sa * (2 * na), general)))
