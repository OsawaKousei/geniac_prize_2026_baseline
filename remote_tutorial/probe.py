"""Reset and render the configured CaP-X simulator without model servers or a trial."""

import sys
from pathlib import Path


def main() -> None:
    import ctypes
    import os

    import numpy as np
    import torch
    import yaml
    from capx.envs import simulators  # noqa: F401 (register the installed simulator)
    from capx.envs.base import get_env
    from PIL import Image

    assert torch.cuda.is_available(), "GPU not available"
    major, minor = ctypes.c_int(), ctypes.c_int()
    library = ctypes.CDLL(os.environ["LD_PRELOAD"])
    assert library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0
    assert (major.value, minor.value) >= (12, 9)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    low = cfg["env"]["cfg"]["low_level"]
    if not isinstance(low, str):
        raise ValueError("initial probe supports named Robosuite low-level simulators")
    # CaP-X ac68084: capx/envs/base.py:100-120 get_env forwards these arguments;
    # capx/envs/simulators/robosuite_base.py:48-53 defines the observation camera,
    # :290-294 closes the simulator. Exercise that simulator, not a replacement task.
    simulator = get_env(low, privileged=False, enable_render=False, viser_debug=False)
    try:
        observation, _ = simulator.reset(seed=301)
        assert observation is not None
        frame = simulator.robosuite_env.sim.render(camera_name=simulator.save_camera_name, width=64, height=64)
        assert frame.shape == (64, 64, 3) and np.isfinite(frame).all()
        Image.fromarray(frame).save(sys.argv[2])
        print("TASK_RESET_RENDER_OK", low, "NVRTC", major.value, minor.value, flush=True)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
