#!/usr/bin/env python3
"""Offscreen MuJoCo render for the E90 report (no GUI). Requires mujoco, imageio."""
from pathlib import Path

import imageio.v2 as imageio
import mujoco

# Repo-relative: hexapod_training_new lives next to final presentation under E90/RL Temp/
_HERE = Path(__file__).resolve().parent
_DEFAULT_XML = _HERE.parent / "RL Temp" / "hexapod_training_new" / "assets" / "hexapod.xml"
_OUT = _HERE / "mujoco_hexapod_sim.png"


def main() -> None:
    xml = Path(_DEFAULT_XML)
    if not xml.is_file():
        raise SystemExit(f"Missing MJCF: {xml}")

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
    print(f"Wrote {_OUT} shape={img.shape}")


if __name__ == "__main__":
    main()
