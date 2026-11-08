"""Hugging Face Space (Gradio): background blur and depth-based lens blur for an uploaded photo.

Space files: app.py, blur_lib.py, requirements.txt   (SDK: Gradio)
Local test:  pip install gradio  &&  python app.py   -> open the printed http://127.0.0.1:7860 link
"""
import gradio as gr
import numpy as np

import blur_lib as bl

# The small depth model keeps the free CPU Space responsive (a few seconds per photo); on a GPU Space you can
# switch to Depth-Anything-V2-Large-hf, as used for the report figures.
SEG_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
segmenter = bl.load_segmenter(SEG_MODEL)
depth_pipe = bl.load_depth_model(DEPTH_MODEL)


def run(image, bg_sigma, max_sigma, mask_aware):
    if image is None:
        raise gr.Error("Upload a photo first.")
    img_pil = bl.load_square(image, 512)
    img = bl.to_float(img_pil)
    mask, _ = bl.person_mask(segmenter, img_pil)
    bg = bl.background_blur(img, mask, sigma=bg_sigma, mask_aware=mask_aware)
    depth = bl.estimate_depth(depth_pipe, img_pil)
    sigma_map = bl.normalize(depth, 0, max_sigma)
    lens = bl.depth_lens_blur(img, sigma_map, max_sigma)
    depth_vis = (255 * bl.normalize(depth, 0, 1)).astype(np.uint8)       # bright = far
    return (np.asarray(img_pil), (mask * 255).astype(np.uint8), bl.to_uint8(bg), depth_vis, bl.to_uint8(lens))


with gr.Blocks(title="Background blur and lens blur") as demo:
    gr.Markdown("# Background blur and depth-based lens blur\n"
                f"Person segmentation: `{SEG_MODEL}` · Depth: `{DEPTH_MODEL}`. "
                "Upload a selfie; it is cropped to 512 x 512.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Your photo")
            bg_sigma = gr.Slider(1, 30, value=15, step=1, label="Background blur sigma (Zoom-style)")
            max_sigma = gr.Slider(1, 30, value=15, step=1, label="Lens blur: sigma at the farthest point")
            aware = gr.Checkbox(value=True, label="Mask-aware blur (no halo around the person)")
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            out_in = gr.Image(label="Input (512 x 512)")
            out_mask = gr.Image(label="Person mask")
    with gr.Row():
        out_bg = gr.Image(label="Background blur")
        out_depth = gr.Image(label="Depth (bright = far)")
        out_lens = gr.Image(label="Depth-based lens blur")
    btn.click(run, [inp, bg_sigma, max_sigma, aware], [out_in, out_mask, out_bg, out_depth, out_lens])

if __name__ == "__main__":
    demo.launch()
