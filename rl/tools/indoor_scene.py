"""Professional visual-only living room shared by Bingo's Isaac viewers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_USD = PROJECT_ROOT / "assets" / "environments" / "bingo_living_room.usda"
RESIDENTIAL_ROOT = PROJECT_ROOT / "assets" / "vendor" / "nvidia_residential"
REQUIRED_ASSET = (
    RESIDENTIAL_ROOT
    / "Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Sofa.usd"
)


def spawn_living_room(sim_utils, root="/World/LivingRoom"):
    """Reference the composed NVIDIA Residential Pack set into the stage.

    The room contains no collision or rigid-body APIs. Stage 4 therefore keeps
    using its validated ground plane and contact model unchanged.
    """

    if not SCENE_USD.is_file() or not REQUIRED_ASSET.is_file():
        raise FileNotFoundError(
            "Bingo's Residential Pack subset is missing. Run:\n"
            "  python3 tools/fetch_residential_assets.py --preset bingo-living-room"
        )
    cfg = sim_utils.UsdFileCfg(usd_path=str(SCENE_USD))
    cfg.func(root, cfg)
    print(f"[[ living-room scene: NVIDIA Residential assets ({SCENE_USD})", flush=True)


def living_room_camera(center):
    """Return a wide, dog-height camera that keeps the furnished room readable."""

    return (
        (float(center[0]) + 1.35, float(center[1]) - 2.05, float(center[2]) + 0.78),
        (float(center[0]), float(center[1]) + 0.18, float(center[2]) + 0.08),
    )
