import numpy as np
import h5py
from torch.utils.data import Dataset
import pytorchvideo.transforms as vid_tvt
import torchvision.transforms as tvt
import torch
import math


class RandomWindowSample(torch.nn.Module):
    def __init__(self, window_size, margin=0):
        super().__init__()
        self.window_size = window_size
        self.margin = margin

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[1] < self.window_size:
            new = torch.zeros(frames.shape[0], self.window_size, *frames.shape[2:])
            new[:, : frames.shape[1]] = frames
            return new

        i = (
            np.random.randint(self.margin, frames.shape[1] - self.margin)
            if frames.shape[1] >= 2 * self.margin + self.window_size
            else (frames.shape[1] - self.window_size) // 2
        )
        return frames[:, i : i + self.window_size]


class CenterWindowSample(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[1] < self.window_size:
            new = torch.zeros(frames.shape[0], self.window_size, *frames.shape[2:])
            new[:, : frames.shape[1]] = frames
            return new

        i = (frames.shape[1] - self.window_size) // 2
        return frames[:, i : i + self.window_size]


class VideoCenterCrop(torch.nn.Module):
    """
    Transform for cropping video frames around the center.
    """

    def __init__(self, crop_size):
        super().__init__()

        self.crop_size = crop_size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        height = frames.shape[2]
        width = frames.shape[3]
        y_offset = int(math.ceil((height - self.crop_size) / 2))
        x_offset = int(math.ceil((width - self.crop_size) / 2))

        return frames[
            :,
            :,
            y_offset : y_offset + self.crop_size,
            x_offset : x_offset + self.crop_size,
        ]


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slowfast_alpha):
        super().__init__()

        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )

        return [slow_pathway, fast_pathway]


class FramesSP(Dataset):
    def __init__(
        self,
        data_fp,
        test=False,
        repeat_frames=True,
        xoffset=20,
        yoffset=20,
        power_mean=215,
        speed_mean=900,
        start_ratio=0.,
        end_ratio=1.0,
        **_
    ):
        self.repeat_frames = repeat_frames
        self.xoffset, self.yoffset = xoffset, yoffset
        self.speed_mean, self.power_mean = speed_mean, power_mean
        self.data_fp = data_fp
        self.test = test
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        # retrieving length
        with h5py.File(self.data_fp, "r") as h5f:
            self.layer_cumsums = [np.cumsum(
                    [
                        len(np.unique(h5f[l]["scan_line_index"][:]))
                        for l in h5f.keys()
                    ]
                )]
            
            self.obj_cumsum = [self.layer_cumsums[-1][-1]]
            self._len = self.obj_cumsum[-1]
            self._len = int(self._len * (self.end_ratio - self.start_ratio))
            

    def __getitem__(self, index):
        # retrieve object, layer, scanline
        index = int(index + self.start_ratio * self._len)
        object_i = 0
        rindex = index if object_i == 0 else index - self.obj_cumsum[object_i - 1]
        layer_i = np.argwhere(self.layer_cumsums[object_i] - 1 - rindex >= 0)[0][0]
        rindex = (
            rindex
            if layer_i == 0
            else rindex - self.layer_cumsums[object_i][layer_i - 1]
        )
        scan_line_i = rindex
        with h5py.File(self.data_fp, "r") as h5f:  # open once ?
            layer = f"layer{layer_i:04d}"
            indices = np.where(h5f[layer]["scan_line_index"][:] == scan_line_i)
            frames = h5f[layer]["frame"][indices]
            if not self.test:
                speed, power = h5f[layer]["laser_params"][scan_line_i]

        # crop frames around max intensity of mean frame
        i, j = np.unravel_index(frames.mean(0).argmax(), frames[0].shape)
        x = tvt.functional.crop(
            torch.tensor(np.array([frames])),
            i - self.xoffset,
            j - self.yoffset,
            2 * self.xoffset + 1,
            2 * self.yoffset + 1,
        )
        if self.repeat_frames:
            x = x.repeat_interleave(3, dim=0)

        y = torch.tensor(
            [
                speed / self.speed_mean,
                power / self.power_mean,
            ]
        )

        return x.float(), y.float()

    def __len__(self):
        return self._len


class OneWaySP(FramesSP):
    """
    Pytorch Dataset to interface the RAISE-LPBF-Laser benchmark data for one-way models like 3DResnet, X3D, MViT, etc.
    """

    def __init__(
        self,
        data_fp,
        mean=0.45,
        std=0.225,
        num_frames=8,
        crop_size=256,
        side_size=256,
        margin=15,
        repeat_frames=True,
        test=False,
        deterministic=False,
        start_ratio=0.,
        end_ratio=1.0,
        **kwargs
    ):
        super().__init__(data_fp, repeat_frames=repeat_frames, test=test, start_ratio=start_ratio, end_ratio=end_ratio, **kwargs)

        _mean = [mean for _ in range(3)] if repeat_frames else [mean]
        _std = [std for _ in range(3)] if repeat_frames else [std]

        self.preprocess = tvt.Compose(
            [
                RandomWindowSample(num_frames, margin=margin),
                vid_tvt.Div255(),
                vid_tvt.Normalize(_mean, _std),
                vid_tvt.ShortSideScale(size=side_size),
                VideoCenterCrop(crop_size),
            ]
        )
        if deterministic:
            # replace first transform with CenterWindowSample
            self.preprocess.transforms[0] = CenterWindowSample(num_frames)

    def __getitem__(self, index):
        frames, y = super().__getitem__(index)

        x = self.preprocess(frames)

        return x, y