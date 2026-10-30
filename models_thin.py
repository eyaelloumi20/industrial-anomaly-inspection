"""
models_thin.py
===============
OPTION 2 — Thinner Student / AutoEncoder architectures for faster inference.

Design decision
---------------
The frozen Teacher is left completely untouched (same architecture, same
weights, same 384-channel output) — the Student/AutoEncoder loss needs the
teacher's output to line up exactly, and retraining the teacher is out of
scope. Instead, we shrink the *hidden* channel widths inside the Student's
PDN blocks and the AutoEncoder's encoder/decoder, controlled by a single
`width_mult` (e.g. 0.5 = half the hidden channels). The input/output channel
counts that must match the Teacher (384 for AE, 384/768 for Student's two
output heads) are left unchanged so the loss functions in retrain_thin.py
work without modification.

This is where most of the compute lives (the hidden 128/256/512-channel
conv layers), so thinning them gives a real FLOPs/latency reduction at the
cost of representational capacity — expect some AUROC drop that needs to be
validated, potentially recovered with longer fine-tuning or a smaller
width_mult reduction (e.g. 0.75 instead of 0.5).

Usage
-----
    from models_thin import Student_Thin, AutoEncoder_Thin
    student = Student_Thin('S', width_mult=0.5)
    ae      = AutoEncoder_Thin(width_mult=0.5)
"""

import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# Allow running this file directly (python models_thin.py) as well as
# importing it from retrain_thin.py in the same directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models import imagenet_norm_batch


def _c(base_channels, width_mult):
    """Scale a channel count by width_mult, keep it a positive multiple of 8
    (helps with hardware-friendly conv shapes) and at least 8."""
    return max(8, int(round(base_channels * width_mult / 8)) * 8)


class PDN_S_Thin(nn.Module):
    """Same topology as PDN_S but hidden widths scaled by width_mult.
    last_kernel_size (the output channel count that must match the Teacher)
    is left untouched."""

    def __init__(self, last_kernel_size=384, width_mult=1.0, with_bn=False) -> None:
        super().__init__()
        self.with_bn = with_bn
        c1 = _c(128, width_mult)
        c2 = _c(256, width_mult)
        c3 = _c(256, width_mult)

        self.conv1 = nn.Conv2d(3,  c1, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(c3, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(c1)
            self.bn2 = nn.BatchNorm2d(c2)
            self.bn3 = nn.BatchNorm2d(c3)
            self.bn4 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        return x


class PDN_M_Thin(nn.Module):
    """Same topology as PDN_M but hidden widths scaled by width_mult."""

    def __init__(self, last_kernel_size=384, width_mult=1.0, with_bn=False) -> None:
        super().__init__()
        self.with_bn = with_bn
        c1 = _c(256, width_mult)
        c2 = _c(512, width_mult)
        c3 = _c(512, width_mult)
        c4 = _c(512, width_mult)

        self.conv1 = nn.Conv2d(3,  c1, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(c4, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(last_kernel_size, last_kernel_size, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(c1)
            self.bn2 = nn.BatchNorm2d(c2)
            self.bn3 = nn.BatchNorm2d(c3)
            self.bn4 = nn.BatchNorm2d(c4)
            self.bn5 = nn.BatchNorm2d(last_kernel_size)
            self.bn6 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x) if self.with_bn else x
        return x


class Student_Thin(nn.Module):
    """Student with thinned hidden widths. Output channel count (768 =
    2 x 384, one head compared to Teacher, one head compared to the AE) is
    unchanged so it still matches OUT_CHANNELS-based loss slicing."""

    def __init__(self, size, width_mult=1.0, with_bn=False, channel_size=768) -> None:
        super().__init__()
        if size == 'M':
            self.pdn = PDN_M_Thin(last_kernel_size=channel_size, width_mult=width_mult, with_bn=with_bn)
        elif size == 'S':
            self.pdn = PDN_S_Thin(last_kernel_size=channel_size, width_mult=width_mult, with_bn=with_bn)
        else:
            raise ValueError(f"size must be 'S' or 'M', got '{size}'")

    def forward(self, x):
        x = imagenet_norm_batch(x)
        return self.pdn(x)


class EncConv_Thin(nn.Module):
    def __init__(self, width_mult=1.0) -> None:
        super().__init__()
        c1 = _c(32, width_mult)
        c2 = _c(32, width_mult)
        c3 = _c(64, width_mult)
        c4 = _c(64, width_mult)
        c5 = _c(64, width_mult)
        c6 = _c(64, width_mult)
        self.enconv1 = nn.Conv2d(3,  c1, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(c4, c5, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(c5, c6, kernel_size=8, stride=1, padding=0)
        self.out_channels = c6

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class DecConv_Thin(nn.Module):
    """Decoder must still emit 384 channels (compared against the frozen
    Teacher), so only the internal 64-wide hidden channels are thinned."""

    def __init__(self, in_channels=64, width_mult=1.0, is_bn=False) -> None:
        super().__init__()
        self.is_bn = is_bn
        h = _c(64, width_mult)
        self.deconv1 = nn.Conv2d(in_channels, h, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(h, h, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(h, h, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(h, h, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(h, h, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(h, h, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(h, h, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(h, 384, kernel_size=3, stride=1, padding=1)

        if is_bn:
            self.dropout1 = nn.BatchNorm2d(h)
            self.dropout2 = nn.BatchNorm2d(h)
            self.dropout3 = nn.BatchNorm2d(h)
            self.dropout4 = nn.BatchNorm2d(h)
            self.dropout5 = nn.BatchNorm2d(h)
            self.dropout6 = nn.BatchNorm2d(h)
        else:
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)
            self.dropout4 = nn.Dropout(p=0.2)
            self.dropout5 = nn.Dropout(p=0.2)
            self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x)); x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode='bilinear')
        x = F.relu(self.deconv2(x)); x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode='bilinear')
        x = F.relu(self.deconv3(x)); x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode='bilinear')
        x = F.relu(self.deconv4(x)); x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode='bilinear')
        x = F.relu(self.deconv5(x)); x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode='bilinear')
        x = F.relu(self.deconv6(x)); x = self.dropout6(x)
        x = F.interpolate(x, size=64, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder_Thin(nn.Module):
    def __init__(self, width_mult=1.0, is_bn=False) -> None:
        super().__init__()
        self.encoder = EncConv_Thin(width_mult=width_mult)
        self.decoder = DecConv_Thin(in_channels=self.encoder.out_channels,
                                     width_mult=width_mult, is_bn=is_bn)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    from utils.models import Student, AutoEncoder
    for wm in (1.0, 0.75, 0.5, 0.25):
        s = Student_Thin('S', width_mult=wm)
        a = AutoEncoder_Thin(width_mult=wm)
        print(f'width_mult={wm:>4}  '
              f'Student params={count_params(s):>9,}  '
              f'AE params={count_params(a):>9,}')
    print(f'--- reference (width_mult=1.0, original modules) ---')
    print(f'Student params={count_params(Student("S")):>9,}  '
          f'AE params={count_params(AutoEncoder()):>9,}')