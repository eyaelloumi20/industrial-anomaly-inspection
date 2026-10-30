"""
models_adaptive.py
===================
A resolution-agnostic AutoEncoder for use with retrain_lowres.py.

Why this exists
----------------
The original AutoEncoder (utils/models.py) is only correct at 256x256 input:

  - EncConv's final layer is `Conv2d(64, 64, kernel_size=8, stride=0, padding=0)`.
    After 5 stride-2 downsampling convs, a 256px input lands at exactly 8x8,
    which is the only size that kernel-8 conv can consume. At 192px it lands
    at 6x6 and at 224px at 7x7 -- both smaller than the kernel, so the conv
    raises a shape error (this is the crash you hit).

  - DecConv's upsampling schedule (3 -> 8 -> 15 -> 32 -> 63 -> 127 -> 64) is
    hardcoded to finish at 64x64, because that's where the Teacher/Student
    PDN's output spatial size lands for a 256px input. At other resolutions
    the PDN's output size is different, so even if the encoder didn't crash,
    the decoder's output wouldn't line up with the Teacher/Student output
    for the loss computation.

Fix
---
  - EncConvAdaptive: inserts `nn.AdaptiveAvgPool2d((8, 8))` right before the
    original kernel-8 conv. At 256px this is a no-op (8x8 -> 8x8), so
    behaviour there is unchanged. At 192px/224px it resamples the 6x6/7x7
    feature map to 8x8 first, so the same conv works everywhere.

  - DecConvAdaptive: takes a `target_size` (the Teacher/Student PDN's actual
    output spatial size for the resolution you're training at) and scales
    the original [3, 8, 15, 32, 63, 127, 64] schedule proportionally,
    guaranteeing the final step lands exactly on `target_size`.

`target_size` should be measured empirically with a dummy forward pass
through the (already-loaded) frozen Teacher -- see `probe_output_size()`
below -- rather than computed by hand, since the PDN's padding/kernel
arithmetic is easy to get subtly wrong.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running this file directly (python models_adaptive.py) as well as
# importing it from retrain_lowres.py in the same directory. Either way,
# the project root (which contains the utils/ package) needs to be on
# sys.path for "from utils.models import ..." to resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models import imagenet_norm_batch


@torch.no_grad()
def probe_output_size(model: nn.Module, image_size: int, device) -> int:
    """Run a single dummy forward pass to measure a fully-convolutional
    model's output spatial size (assumed square) for a given input size."""
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    out = model(dummy)
    model.train(was_training)
    return int(out.shape[-1])


class EncConvAdaptive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3,  32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        # Normalise to 8x8 before the bottleneck conv, regardless of the
        # actual feature-map size at this point (8x8 at 256px input, 7x7 at
        # 224px, 6x6 at 192px). No-op at 256px.
        self.pre_bottleneck_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.pre_bottleneck_pool(x)
        x = self.enconv6(x)
        return x


def _scaled_schedule(target_size: int):
    """Scale the original [3, 8, 15, 32, 63, 127] growth schedule
    proportionally to target_size/64, keeping it strictly increasing, then
    append target_size as the final (possibly downsizing) step -- mirrors
    the original network's behaviour of overshooting to 127 before
    settling back down to its final 64."""
    base = [3, 8, 15, 32, 63, 127]
    scale = target_size / 64.0
    scaled, prev = [], 0
    for b in base:
        v = max(prev + 1, round(b * scale))
        scaled.append(v)
        prev = v
    return scaled + [target_size]


class DecConvAdaptive(nn.Module):
    def __init__(self, target_size: int, is_bn=False) -> None:
        super().__init__()
        self.is_bn = is_bn
        self.schedule = _scaled_schedule(target_size)  # 7 sizes, last == target_size

        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)

        if is_bn:
            self.dropout1 = nn.BatchNorm2d(64)
            self.dropout2 = nn.BatchNorm2d(64)
            self.dropout3 = nn.BatchNorm2d(64)
            self.dropout4 = nn.BatchNorm2d(64)
            self.dropout5 = nn.BatchNorm2d(64)
            self.dropout6 = nn.BatchNorm2d(64)
        else:
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)
            self.dropout4 = nn.Dropout(p=0.2)
            self.dropout5 = nn.Dropout(p=0.2)
            self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        s = self.schedule
        x = F.interpolate(x, size=s[0], mode='bilinear')
        x = F.relu(self.deconv1(x)); x = self.dropout1(x)
        x = F.interpolate(x, size=s[1], mode='bilinear')
        x = F.relu(self.deconv2(x)); x = self.dropout2(x)
        x = F.interpolate(x, size=s[2], mode='bilinear')
        x = F.relu(self.deconv3(x)); x = self.dropout3(x)
        x = F.interpolate(x, size=s[3], mode='bilinear')
        x = F.relu(self.deconv4(x)); x = self.dropout4(x)
        x = F.interpolate(x, size=s[4], mode='bilinear')
        x = F.relu(self.deconv5(x)); x = self.dropout5(x)
        x = F.interpolate(x, size=s[5], mode='bilinear')
        x = F.relu(self.deconv6(x)); x = self.dropout6(x)
        x = F.interpolate(x, size=s[6], mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoderAdaptive(nn.Module):
    def __init__(self, target_size: int, is_bn=False) -> None:
        super().__init__()
        self.encoder = EncConvAdaptive()
        self.decoder = DecConvAdaptive(target_size=target_size, is_bn=is_bn)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # quick self-test across all three resolutions
    import torch
    from utils.models import Teacher

    for size in (192, 224, 256):
        teacher = Teacher('S').eval()
        t_out_size = probe_output_size(teacher, size, device='cpu')
        ae = AutoEncoderAdaptive(target_size=t_out_size)
        dummy = torch.zeros(1, 3, size, size)
        out = ae(dummy)
        print(f'image_size={size:3d}  teacher_out={t_out_size}x{t_out_size}  '
              f'ae_out={tuple(out.shape)}  schedule={ae.decoder.schedule}')
        assert out.shape[-1] == t_out_size, 'AE output size must match Teacher/Student output size'
    print('OK: AutoEncoderAdaptive output matches Teacher output size at all three resolutions.')