# Problem 3: Selfie segmentation, background blur, depth and lens blur

## The pipeline

1. **Pre-process:** open the selfie, apply the phone's EXIF rotation (photos from phones are often stored
   sideways), resize so the shorter side is 512, center-crop to 512x512.
2. **Segment** with a transformer from the Hugging Face Hub. The default is Mask2Former with a Swin-L
   backbone, trained on COCO panoptic (`facebook/mask2former-swin-large-coco-panoptic`). It returns one
   mask per detected thing; we keep the `person` masks. The person is white (1), the background black (0).
   Holes are filled, and only the largest connected region is kept, so people far behind you are ignored.
3. **Background blur:** Gaussian blur with sigma = 15 on the background only, the person stays sharp.
4. **Depth** with Depth Anything V2 (`depth-anything/Depth-Anything-V2-Large-hf`).
5. **Lens blur:** depth normalized to [0, 15] becomes a per-pixel blur sigma, so far things are blurred more.

## Two details worth explaining in the report

**Depth models output inverse depth.** Depth Anything (and DPT/MiDaS) predict relative *disparity*:
large values mean close. The assignment wants large = far, so `estimate_depth()` flips it
(`max - d`). If you skip this, the lens blur blurs YOU and keeps the background sharp. That is an easy
way to check it worked.

**The "halo" problem with a naive background blur.** If you blur the whole image and paste the sharp
person on top, the blurred background near your outline contains smeared-out person colors, which shows
as a ghostly halo. `background_blur(..., mask_aware=True)` blurs only background pixels and
renormalizes: `blur(image * bg) / blur(bg)`. `fig5_halo_comparison.png` shows both versions zoomed on
your outline. This is a nice extra figure.

**Variable blur efficiently.** A different Gaussian kernel for each of 262,144 pixels would be slow.
Instead we blur the whole image at sigma = 0, 1, 2, ..., 15 once (16 images). Each pixel then takes a
linear mix of the two images around its own sigma (for sigma = 6.3: 70% of the sigma-6 image + 30% of the
sigma-7 image).

## Steps

1. Take a selfie with a busy background (not a plain wall). Copy it to `p3_segmentation/selfie.jpg`.
2. `sbatch run_p3.sbatch`. Or interactively: `python run_p3.py --image selfie.jpg --out results`.
   The first run downloads the two models (~1.5 GB) into `~/.cache/huggingface`.
3. Check `results/`:
   - `fig1_segmentation.png`: input and mask (part 1)
   - `fig2_background_blur.png`: input and blurred background (part 2)
   - `fig3_depth.png`: input and depth map, bright = far (part 3)
   - `fig4_lens_blur.png`: input, blur map, lens-blurred result (part 4); `fig4b_lens_blur_pair.png` is
     just input and result, side by side
   - `fig5_halo_comparison.png`: naive vs. mask-aware blur (extra)

If the mask misses part of you (hair, a hand), try the other models with `--seg-model`:
`nvidia/segformer-b5-finetuned-ade-640-640` (SegFormer, ADE20K, label "person") or
`mattmdjaga/segformer_b2_clothes` (human parsing: hair, face, clothes, ... all combined).
The code handles both label styles.

## Extra credit

**(a) Colab notebook.** `HW3_P3_colab.ipynb` is already generated (rebuild with `python make_colab.py`
after changing `blur_lib.py`). It contains the library code, installs what it needs, and runs all four parts.
1. Put your selfie somewhere public: a GitHub repo (use the "raw" link) or a Google Drive file shared
   with "Anyone with the link" (use `https://drive.google.com/uc?id=<FILE_ID>`).
2. Upload the notebook to Colab (File > Upload notebook), set `IMAGE_URL`, Runtime > Run all.
3. Share it: Share > "Anyone with the link can view", and put the link in the report.

**(b) Hugging Face Space.**
1. Create a free account on huggingface.co, then New > Space. SDK: **Gradio**, hardware: free CPU.
2. Upload `app.py`, `blur_lib.py` and `requirements.txt` (Files > Add file > Upload files).
3. Wait for the build (~5 min; watch the "Logs" tab), then upload your selfie and take a screenshot of
   the result. Put the Space URL and the screenshot in the report.
The app has sliders for both blur strengths and a mask-aware on/off switch, so the user input changes the
result, which is what the extra credit asks for. To test locally first: `pip install gradio && python app.py`.

## For the report

- Which models you chose and why (both are transformers: Mask2Former/Swin, Depth Anything V2/ViT), with
  citations: Cheng et al. CVPR 2022 (Mask2Former), Yang et al. NeurIPS 2024 (Depth Anything V2).
- The four figures, each with a caption.
- The two technical points above (inverse depth, halo), and the formula of the variable blur.
- Observed failure cases (hair edges, thin objects, glass), if any.
