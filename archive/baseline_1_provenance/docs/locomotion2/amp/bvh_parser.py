"""Generic BVH parser + vectorized forward kinematics (pure numpy, no external bvh lib).

The lifelike_dog clips use a non-standard-but-legal BVH dialect: EVERY joint (not
just the root) declares 6 channels (Xposition Yposition Zposition Zrotation
Xrotation Yrotation), a pattern from 3ds-Max-Biped-style exporters where a human
biped rig has been retargeted onto a quadruped mesh (bone names like b_LeftArm /
b_LeftForeArm / b_LeftHand for the front legs, b_LeftLegUpper / b_LeftLeg /
b_LeftLeg1 / b_LeftAnkle / b_LeftToe for the hind legs). This parser makes no
assumption about which joints carry which channels -- it reads each joint's own
CHANNELS line and uses per-frame position values where present, falling back to
the static OFFSET otherwise, and composes rotations in the exact order listed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

POS_CHANNELS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
ROT_AXIS = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


@dataclass
class Joint:
    name: str
    parent: int  # index into joints list, -1 for root
    offset: np.ndarray  # (3,)
    channels: list  # list of channel name strings, in file order
    is_end_site: bool = False
    children: list = field(default_factory=list)


@dataclass
class BVH:
    joints: list  # list[Joint]
    motion: np.ndarray  # (T, C) raw channel values, C = total channels across all joints (end sites excluded)
    frame_time: float
    channel_offsets: list  # per-joint start index into motion's column axis (-1 for end sites, no channels)

    @property
    def n_frames(self):
        return self.motion.shape[0]

    @property
    def fps(self):
        return 1.0 / self.frame_time

    def index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(f"joint '{name}' not found")


_TOKEN_RE = re.compile(r"\S+")


def parse_bvh(path: str) -> BVH:
    with open(path, "r") as f:
        text = f.read()

    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    assert lines[i] == "HIERARCHY"
    i += 1

    joints: list[Joint] = []
    channel_offsets: list[int] = []
    chan_cursor = 0
    stack = []  # (parent_index,)

    def parse_joint(i, parent_idx, is_end=False, name_override=None):
        nonlocal chan_cursor
        # lines[i] is "ROOT name" / "JOINT name" / "End Site"
        toks = _TOKEN_RE.findall(lines[i])
        if toks[0] == "End":
            name = f"{name_override}_End"
        else:
            name = toks[1]
        i += 1
        assert lines[i] == "{"
        i += 1
        offset = np.zeros(3)
        chans: list[str] = []
        my_idx = len(joints)
        joints.append(Joint(name=name, parent=parent_idx, offset=offset, channels=chans, is_end_site=is_end))
        if parent_idx >= 0:
            joints[parent_idx].children.append(my_idx)
        while True:
            line = lines[i]
            if line.startswith("OFFSET"):
                vals = [float(v) for v in _TOKEN_RE.findall(line)[1:]]
                joints[my_idx].offset = np.array(vals)
                i += 1
            elif line.startswith("CHANNELS"):
                toks = _TOKEN_RE.findall(line)
                n = int(toks[1])
                names = toks[2:2 + n]
                joints[my_idx].channels = names
                channel_offsets.append(chan_cursor)
                chan_cursor += n
                i += 1
            elif line.startswith("JOINT"):
                i = parse_joint(i, my_idx)
            elif line.startswith("End Site"):
                channel_offsets.append(-1)
                i = parse_joint(i, my_idx, is_end=True, name_override=joints[my_idx].name)
            elif line == "}":
                i += 1
                break
            else:
                raise ValueError(f"unexpected line in HIERARCHY: {line!r}")
        return i

    i = parse_joint(i, -1)

    while lines[i] == "":
        i += 1
    assert lines[i] == "MOTION", lines[i]
    i += 1
    n_frames = int(_TOKEN_RE.findall(lines[i])[1])
    i += 1
    frame_time = float(_TOKEN_RE.findall(lines[i])[2])
    i += 1
    data = np.zeros((n_frames, chan_cursor), dtype=np.float64)
    for f in range(n_frames):
        vals = [float(v) for v in lines[i + f].split()]
        data[f, :] = vals
    assert len(channel_offsets) == len(joints)

    return BVH(joints=joints, motion=data, frame_time=frame_time, channel_offsets=channel_offsets)


def _euler_axis_mat(axis: str, angle_deg: np.ndarray) -> np.ndarray:
    """angle_deg: (...,) -> (...,3,3) rotation matrices about the given local axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    zeros = np.zeros_like(a)
    ones = np.ones_like(a)
    if axis == "x":
        R = np.stack([
            ones, zeros, zeros,
            zeros, c, -s,
            zeros, s, c,
        ], axis=-1)
    elif axis == "y":
        R = np.stack([
            c, zeros, s,
            zeros, ones, zeros,
            -s, zeros, c,
        ], axis=-1)
    elif axis == "z":
        R = np.stack([
            c, -s, zeros,
            s, c, zeros,
            zeros, zeros, ones,
        ], axis=-1)
    else:
        raise ValueError(axis)
    return R.reshape(a.shape + (3, 3))


def forward_kinematics(bvh: BVH, scale: float = 1.0):
    """Vectorized FK across all frames.

    Returns:
      pos:  (T, J, 3) world positions of every joint/end-site, in the SAME units
            as the BVH file times `scale` (pass scale=0.01 for cm->m files).
      rot:  (T, J, 3, 3) world rotation matrices (end sites inherit their parent's).
    """
    T = bvh.n_frames
    J = len(bvh.joints)
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 3, 3))
    ident = np.eye(3)

    for j_idx, joint in enumerate(bvh.joints):
        off = joint.offset * scale
        if joint.is_end_site:
            local_t = np.broadcast_to(off, (T, 3)).copy()
            local_R = np.broadcast_to(ident, (T, 3, 3)).copy()
        else:
            start = bvh.channel_offsets[j_idx]
            chans = joint.channels
            vals = bvh.motion[:, start:start + len(chans)]
            local_t = np.broadcast_to(off, (T, 3)).copy()
            for ci, cname in enumerate(chans):
                if cname in POS_CHANNELS:
                    local_t[:, POS_CHANNELS[cname]] = vals[:, ci] * scale
            local_R = np.broadcast_to(ident, (T, 3, 3)).copy()
            for ci, cname in enumerate(chans):
                if cname in ROT_AXIS:
                    Rc = _euler_axis_mat(ROT_AXIS[cname], vals[:, ci])
                    local_R = np.einsum("tij,tjk->tik", local_R, Rc)

        if joint.parent < 0:
            pos[:, j_idx] = local_t
            rot[:, j_idx] = local_R
        else:
            p_pos = pos[:, joint.parent]
            p_rot = rot[:, joint.parent]
            pos[:, j_idx] = p_pos + np.einsum("tij,tj->ti", p_rot, local_t)
            rot[:, j_idx] = np.einsum("tij,tjk->tik", p_rot, local_R)

    return pos, rot


def detect_scale(bvh: BVH) -> float:
    """Heuristic: lifelike_dog offsets are in cm (root height ~50-60 units).
    A real dog is 0.3-0.8 m tall; if the raw skeleton height is >5 (i.e. clearly
    not already meters), assume centimeters and return 0.01, else 1.0."""
    root = bvh.joints[0]
    h = abs(root.offset[1]) + abs(root.offset[2])  # Y or Z might be "up" depending on rig
    # more robust: use frame-0 root position magnitude directly
    start = bvh.channel_offsets[0]
    chans = bvh.joints[0].channels
    vals = bvh.motion[0, start:start + len(chans)]
    pos_vals = [vals[ci] for ci, c in enumerate(chans) if c in POS_CHANNELS]
    mag = max(abs(v) for v in pos_vals) if pos_vals else h
    return 0.01 if mag > 5.0 else 1.0


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/pub0/muhammadf/Blender/dataset/lifelike_dog/raw_bvh/raw_bvh_data/dog_quad_walk_001.bvh"
    b = parse_bvh(p)
    print(f"{p}: {len(b.joints)} joints, {b.n_frames} frames @ {b.fps:.1f} fps")
    scale = detect_scale(b)
    print(f"detected scale factor: {scale}")
    pos, rot = forward_kinematics(b, scale=scale)
    print("pos shape", pos.shape)
    names = [j.name for j in b.joints]
    for n in ["Bip01", "b_Hips", "b_LeftHand", "b_RightHand", "b_LeftToe", "b_RightToe"]:
        if n in names:
            idx = names.index(n)
            print(f"{n:16s} frame0 pos = {pos[0, idx]}")
