"""Convert v4 URDF -> USD using the isaacsim URDF importer command API directly
(bypasses IsaacLab's UrdfConverter, which is incompatible with this importer version).
"""
import argparse, os, re
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.commands, omni.usd
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")
from pxr import Usd


def main():
    v4dir = os.path.dirname(os.path.dirname(os.path.abspath(args.urdf)))
    with open(args.urdf) as f:
        txt = f.read()
    txt = re.sub(r"package://[^/]+/", v4dir + "/", txt)
    tmp_urdf = os.path.join(v4dir, "urdf", "_v4_abs_tmp.urdf")
    with open(tmp_urdf, "w") as f:
        f.write(txt)

    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    for attr, val in [
        ("merge_fixed_joints", False), ("fix_base", False), ("import_inertia_tensor", True),
        ("distance_scale", 1.0), ("convex_decomp", False), ("self_collision", False),
        ("make_default_prim", True), ("create_physics_scene", False),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
    print("import_config set", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # dest_path writes a clean asset USD directly
    res = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=tmp_urdf, import_config=cfg,
        dest_path=args.out, get_articulation_root=True,
    )
    print("import result:", res, flush=True)

    if not os.path.exists(args.out):
        # fallback: export the current stage
        stage = omni.usd.get_context().get_stage()
        stage.Export(args.out)
    os.remove(tmp_urdf)

    if os.path.exists(args.out):
        st = Usd.Stage.Open(args.out)
        joints = [p for p in st.Traverse() if "Joint" in p.GetTypeName() and "Fixed" not in p.GetTypeName()]
        names = [p.GetName() for p in joints]
        print("USD_JOINTS", len(names), flush=True)
        print("EAR_JOINTS", [n for n in names if "ear" in n.lower()], flush=True)
        print("CONVERT_OK", args.out, flush=True)
    else:
        print("CONVERT_FAILED no usd", flush=True)


_ok = True
try:
    main()
except Exception:
    import traceback; traceback.print_exc(); _ok = False
os._exit(0 if _ok else 1)
