#!/usr/bin/env python3
"""List or selectively extract files from NVIDIA's Residential Assets Pack.

The official archive is 22.5 GB.  This utility reads its ZIP central directory
with HTTP byte ranges and downloads only requested members, which keeps the
project usable on machines that cannot store the complete pack.
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
from pathlib import Path
import re
import struct
import sys
import urllib.request
import zlib


ARCHIVE_URL = "https://d4i3qtqj3r0z5.cloudfront.net/Residential_NVD%4010012.zip"
INDEX_NAME = ".residential_archive_index.json"
BINGO_LIVING_ROOM_PATTERNS = (
    r"^Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_(Chair|Sofa|Loveseat|EndTable|SofaTable_Var01)\.usd$",
    r"^Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Textures/Appleseed_(Fabric01|MetalBase|Pillow01|Pillow02|EndTable_|SofaTable_).+\.png$",
    r"^Assets/ArchVis/Residential/Decor/Rugs/(Blue_Rugs\.usd|Textures/BlueRug_.+\.png)$",
    r"^Assets/ArchVis/Residential/Furniture/MediaTables/(Manchester\.usd|Textures/Manchester_.+\.png)$",
    r"^Assets/ArchVis/Residential/Electronics/Televisions/(TV_Wall\.usd|textures/TV_Wall_.+\.tga)$",
    r"^Assets/ArchVis/Residential/Plants/(Plant_0[12]\.usd|Textures/(Plant_dirt_1(_bump)?\.jpg|marble\.png|plant0[12]_leaves_.+\.png))$",
    r"^Assets/ArchVis/Residential/Lighting/Table Lamps/(Mechant\.usd|textures/(Brass_.+|GradientBlueYellow|Linen_Beige_.+|SoftNoise_.+)\.png)$",
    r"^Assets/ArchVis/Residential/Decor/Magazines/(MagazineStack01\.usd|textures/Mag.+\.tga)$",
    r"^Assets/ArchVis/Residential/Decor/Pictures/(PictureFrame02\.usd|textures/(P_Kirsi_01\.jpg|PictureFrame_.+\.tga))$",
)


def request_range(url: str, start: int, end: int):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206:
        response.close()
        raise RuntimeError(f"Server ignored byte range {start}-{end} (HTTP {response.status})")
    return response


def read_range(url: str, start: int, end: int) -> bytes:
    with request_range(url, start, end) as response:
        return response.read()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def zip64_value(extra: bytes, usize: int, csize: int, offset: int, disk: int):
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        data = extra[cursor + 4 : cursor + 4 + size]
        cursor += 4 + size
        if kind != 1:
            continue
        pos = 0
        if usize == 0xFFFFFFFF:
            usize = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        if csize == 0xFFFFFFFF:
            csize = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
        if disk == 0xFFFF:
            disk = struct.unpack_from("<I", data, pos)[0]
        break
    return usize, csize, offset


def fetch_index(url: str) -> list[dict]:
    size = remote_size(url)
    tail_size = min(size, 131072)
    tail = read_range(url, size - tail_size, size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")

    locator_pos = tail.rfind(b"PK\x06\x07", 0, eocd_pos)
    if locator_pos >= 0:
        _, _, zip64_offset, _ = struct.unpack_from("<4sIQI", tail, locator_pos)
        record = read_range(url, zip64_offset, zip64_offset + 55)
        values = struct.unpack_from("<4sQ2H2I4Q", record)
        entry_count, directory_size, directory_offset = values[7], values[8], values[9]
    else:
        values = struct.unpack_from("<4s4H2IH", tail, eocd_pos)
        entry_count, directory_size, directory_offset = values[4], values[5], values[6]

    directory = read_range(url, directory_offset, directory_offset + directory_size - 1)
    entries: list[dict] = []
    cursor = 0
    while cursor < len(directory):
        values = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
        if values[0] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid ZIP central-directory entry at byte {cursor}")
        flag, method = values[3], values[4]
        crc, csize, usize = values[7], values[8], values[9]
        name_len, extra_len, comment_len = values[10], values[11], values[12]
        disk, offset = values[13], values[16]
        begin = cursor + 46
        encoded_name = directory[begin : begin + name_len]
        name = encoded_name.decode("utf-8" if flag & 0x800 else "cp437")
        extra = directory[begin + name_len : begin + name_len + extra_len]
        usize, csize, offset = zip64_value(extra, usize, csize, offset, disk)
        entries.append(
            {
                "name": name,
                "compressed_size": csize,
                "size": usize,
                "method": method,
                "offset": offset,
                "crc32": crc,
            }
        )
        cursor += 46 + name_len + extra_len + comment_len

    if len(entries) != entry_count:
        raise RuntimeError(f"Expected {entry_count} ZIP members, parsed {len(entries)}")
    return entries


def load_index(url: str, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    entries = fetch_index(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def extract_entry(url: str, entry: dict, output_root: Path) -> None:
    name = entry["name"]
    destination = output_root / name
    if name.endswith("/"):
        destination.mkdir(parents=True, exist_ok=True)
        return
    if destination.exists() and destination.stat().st_size == entry["size"]:
        print(f"skip {name}")
        return

    header = read_range(url, entry["offset"], entry["offset"] + 29)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {name}")
    name_len, extra_len = values[-2], values[-1]
    data_start = entry["offset"] + 30 + name_len + extra_len
    data_end = data_start + entry["compressed_size"] - 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    decompressor = zlib.decompressobj(-15) if entry["method"] == 8 else None
    if entry["method"] not in (0, 8):
        raise RuntimeError(f"Unsupported ZIP compression method {entry['method']} for {name}")

    crc = 0
    written = 0
    with request_range(url, data_start, data_end) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            decoded = decompressor.decompress(block) if decompressor else block
            if decoded:
                output.write(decoded)
                crc = binascii.crc32(decoded, crc)
                written += len(decoded)
        if decompressor:
            decoded = decompressor.flush()
            output.write(decoded)
            crc = binascii.crc32(decoded, crc)
            written += len(decoded)

    if written != entry["size"] or crc & 0xFFFFFFFF != entry["crc32"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Integrity check failed for {name}")
    os.replace(temporary, destination)
    print(f"fetched {name} ({written / 1024**2:.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/vendor/nvidia_residential"))
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--list", action="store_true", help="list matching members without downloading")
    parser.add_argument("--match", action="append", default=[], help="case-insensitive regex; repeatable")
    parser.add_argument("--prefix", action="append", default=[], help="archive path prefix; repeatable")
    parser.add_argument("--preset", choices=("bingo-living-room",), help="download a verified asset subset")
    args = parser.parse_args()

    cache_path = args.output / INDEX_NAME
    entries = load_index(args.url, cache_path)
    raw_patterns = list(args.match)
    if args.preset == "bingo-living-room":
        raw_patterns.extend(BINGO_LIVING_ROOM_PATTERNS)
    patterns = [re.compile(value, re.IGNORECASE) for value in raw_patterns]
    selected = [
        entry
        for entry in entries
        if (not patterns and not args.prefix)
        or any(pattern.search(entry["name"]) for pattern in patterns)
        or any(entry["name"].startswith(prefix) for prefix in args.prefix)
    ]
    if not selected:
        print("No matching archive members", file=sys.stderr)
        return 1

    total = sum(entry["size"] for entry in selected if not entry["name"].endswith("/"))
    print(f"selected {len(selected)} entries, {total / 1024**2:.1f} MiB uncompressed")
    if args.list:
        for entry in selected:
            print(f"{entry['size']:12d}  {entry['name']}")
        return 0

    for entry in selected:
        extract_entry(args.url, entry, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
