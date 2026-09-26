"""Single-image wrapper around the DAATSim atmospheric turbulence simulator.

DAATSim (https://github.com/Riponcs/DAATSim, Saha et al., Pacific Graphics 2025)
ships as a script (main.py) that renders a video from one image. This module
loads only the helper functions from main.py (everything above the script's
"Output Directory Setup" section) and reproduces one frame of its tilt + blur
pipeline, in uniform (no-depth) mode, so we can apply turbulence to many images.

Usage:
    sim = DAATSimTurbulence("path/to/DAATSim", size=256)
    turbulent = sim(clean_chw_tensor_in_0_1, r0=0.02)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

_CUT_MARKER = "# --- Output Directory Setup ---"


def _load_daatsim_helpers(daatsim_dir, device):
    daatsim_dir = Path(daatsim_dir).resolve()
    src = (daatsim_dir / "main.py").read_text()
    if _CUT_MARKER not in src:
        raise RuntimeError(f"Could not find '{_CUT_MARKER}' in {daatsim_dir / 'main.py'}; "
                           "the DAATSim code layout may have changed.")
    src = src.split(_CUT_MARKER)[0]
    # The depth-estimation import pulls in `transformers`; uniform mode doesn't need it.
    src = src.replace("from MetDepthFusion import DepthFusionPipeline", "")
    sys.path.insert(0, str(daatsim_dir))          # so `import config` finds DAATSim's config.py
    try:
        import config  # noqa: F401  (DAATSim's config module)
        config.DEVICE_OVERRIDE = str(device)
        ns = {"__name__": "daatsim_main"}
        exec(compile(src, str(daatsim_dir / "main.py"), "exec"), ns)
    finally:
        sys.path.remove(str(daatsim_dir))
    return ns, config


class DAATSimTurbulence:
    """Applies one random realization of DAATSim turbulence (tilt + blur) to an image."""

    def __init__(self, daatsim_dir, size=256, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ns, self.cfg = _load_daatsim_helpers(daatsim_dir, self.device)
        self.N = size
        c = self.cfg
        self.delta0 = c.L_PROP_PARAM * c.WAVELENGTH_PARAM / (2 * c.D_APERTURE_PARAM)
        self.scaling = c.OBJ_SIZE_PARAM / (size * self.delta0)
        self.num_levels = c.NUM_PSF_LEVELS_LOW_RES if size > 512 else c.NUM_PSF_LEVELS_HIGH_RES
        grid, norm = self.ns["compute_grid"](size, size, self.device)
        self.base_grid = (grid / norm.view(1, 1, 2)) * 2.0 - 1.0
        self._per_r0 = {}

    def _setup_r0(self, r0):
        """Everything main.py precomputes once per r0 value (patch layout, blur masks)."""
        if r0 in self._per_r0:
            return self._per_r0[r0]
        c, N = self.cfg, self.N
        Dr0 = c.D_APERTURE_PARAM / r0
        lo, hi = Dr0 * c.MIN_BLUR_REL_STRENGTH_PARAM, Dr0 * c.MAX_BLUR_REL_STRENGTH_PARAM
        smax = (self.delta0 / c.D_APERTURE_PARAM * N) * self.scaling
        k = np.arange(1, 101)
        patchN = k[np.argmin((smax / k - 2) ** 2)]
        patch_size = max(1, min(round(N / patchN), N))
        num_patches = max(1, N // patch_size)
        masks = self.ns["precompute_blur_data"]((N, N), patch_size, num_patches, self.device)
        setup = dict(lo=lo, hi=hi, patch_size=patch_size, P=num_patches ** 2, masks=masks)
        self._per_r0[r0] = setup
        return setup

    @torch.no_grad()
    def __call__(self, image, r0):
        """image: 3 x N x N float tensor in [0, 1]. Returns a turbulent image, same shape."""
        c, ns, N = self.cfg, self.ns, self.N
        s = self._setup_r0(float(r0))
        image = image.to(self.device).float()

        # 1) Tilt: warp the image with the gradient of a random phase screen.
        phase = ns["generate_phase_screen"](N, r0, c.L0_PARAM, c.l0_PARAM, c.OBJ_SIZE_PARAM,
                                            self.device, scale_factor=c.PHASE_SCREEN_SCALE_FACTOR_PARAM)
        dy, dx = torch.gradient(phase)
        to_disp = c.WAVELENGTH_PARAM / (2 * math.pi) * c.FOCAL_LENGTH_PARAM * 2.0 / c.OBJ_SIZE_PARAM
        gain = c.TILT_SCALE_FACTOR_PARAM * (N / 128) ** 1.8
        disp = torch.stack([dx * to_disp * gain, dy * to_disp * gain], dim=-1)
        grid = torch.clamp(self.base_grid + disp, -1.0, 1.0)
        tilted = torch.nn.functional.grid_sample(image[None], grid[None], mode="bilinear",
                                                 padding_mode="reflection", align_corners=True)[0]

        # 2) Blur: spatially varying Zernike PSFs, one set of coefficients per patch.
        Dr0_base = (s["lo"] + s["hi"]) / 2.0
        if c.USE_CORRELATED_COEFFS_PARAM:
            coeffs = ns["genZernikeCoeff_correlated"](
                c.NUM_ZERN_PARAM, Dr0_base, L0_val=c.L0_PARAM, l0_val=c.l0_PARAM,
                D_aperture_val=c.D_APERTURE_PARAM, wavelength_val=c.WAVELENGTH_PARAM,
                L_prop_val=c.L_PROP_PARAM, batch_size=s["P"], device_val=self.device)
        else:
            coeffs = ns["genZernikeCoeff_independent"](c.NUM_ZERN_PARAM, Dr0_base,
                                                       batch_size=s["P"], device_val=self.device)
        uniform_depth = torch.ones(N, N, device=self.device)
        out = ns["fast_blur_image_single_optimized"](
            image=tilted, depth_map=uniform_depth, smoothed_masks=s["masks"],
            coeffs_ref_patches_input=coeffs, patch_size_arg=s["patch_size"],
            scaling_arg=self.scaling, nn_base_arg=c.NN_BASE_PARAM,
            num_psf_levels=self.num_levels, num_zern_arg=c.NUM_ZERN_PARAM,
            min_Dr0_arg=s["lo"], max_Dr0_arg=s["hi"], L0_arg=c.L0_PARAM, l0_arg=c.l0_PARAM,
            D_aperture_arg=c.D_APERTURE_PARAM, wavelength_arg=c.WAVELENGTH_PARAM,
            useDepth_arg=False, L_prop_arg=c.L_PROP_PARAM)
        return out.clamp(0, 1)
