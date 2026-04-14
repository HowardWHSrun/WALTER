#!/usr/bin/env python3
"""Offscreen MuJoCo render for the E90 report (no GUI). Requires mujoco, imageio."""
from pathlib import Path

import imageio.v2 as imageio
import mujoco

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _find_hexapod_xml() -> Path:
    """Public repo: hexapod_training_new/ at repo root. Local E90 tree: RL Temp/hexapod_training_new/."""
    candidates = [
        _ROOT / "hexapod_training_new" / "assets" / "hexapod.xml",
        _ROOT / "RL Temp" / "hexapod_training_new" / "assets" / "hexapod.xml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "hexapod.xml not found. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


_OUT = _HERE / "mujoco_hexapod_sim.png"


def main() -> None:
    xml = _find_hexapod_xml()

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for _ in range(4000):
        mujoco.mj_step(model, data)

    from mujoco import Renderer

    renderer = Renderer(model, height=640, width=960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.06]
    cam.distance = 0.65
    cam.azimuth = 135.0
    cam.elevation = -25.0
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    imageio.imwrite(str(_OUT), img)
    renderer.close()
    print(f"Wrote {_OUT} shape={img.shape} (MJCF {xml})")


if __name__ == "__main__":
    main()
