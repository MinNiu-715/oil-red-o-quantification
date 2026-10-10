# Oil Red O lipid droplet quantification

A small Python script that segments Oil Red O-positive lipid droplets in liver sections and in
stained cultured cells, and reports the lipid droplet area fraction for each image.

This code was used to produce the Oil Red O quantification reported in:

> Niu M, Zhang W, Fan Y, et al. Sleeve gastrectomy ameliorates MASLD in male mice via the gut
> microbiota–3-oxo-LCA–Th17/Treg axis. *(update with the final journal citation)*

## Method

Lipid droplets are segmented with a **fixed absolute threshold on the red-channel excess**,
`R − max(G, B) > 40` (0–255 RGB units), applied uniformly to every image in a batch so that all
groups are processed with an identical threshold.

Per-image adaptive thresholding (Otsu) and colour clustering are deliberately **not** used:
because they adapt to the histogram of each individual image, they misclassify pale
haematoxylin-stained cytoplasm as lipid.

The binary mask is cleaned by a 3 × 3 morphological closing followed by an opening, and connected
components smaller than `--min-area` pixels are discarded. Droplet number and the mean, median
and maximum droplet area are obtained by connected-component analysis. The primary denominator is
the **whole image area**; the fraction relative to the stained tissue area (blank slide background
excluded) is also reported as a reference.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_oro.py --input <image_or_folder> --out <output_folder> \
    --mode fixed --thr 40 --min-area 8 --group-from-name
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | – | A single image or a folder (searched recursively) |
| `--out` | – | Output folder |
| `--mode` | `fixed` | `fixed` = absolute threshold on `R − max(G, B)`. The other modes (`red`, `a`, `sat`, `mix`) use per-image Otsu and are intended for single-image tuning only — do not use them for between-group comparison |
| `--thr` | 40 | Threshold used in `fixed` mode |
| `--min-area` | 20 (use **8** for 40× microscopy) | Minimum connected-component area in pixels |
| `--watershed` | off | Split clumped droplets (requires `scipy` and `scikit-image`) |
| `--group-from-name` | off | Infer groups from file names |

On Windows the wrapper `run_oro.py` is recommended: running `oro_analyze.py` directly through
Git Bash can exit silently without output.

## Outputs

| File | Content |
|---|---|
| `metrics.csv` | Per-image measurements (areas, droplet counts, diameters, area fractions) |
| `summary.png` | Bar / box / droplet-count summary figure |
| `report.html` | Full report with parameter list and QC panels |
| `panel_*.png` | Per-image QC panels: original, red-channel excess, binary mask, overlay |

**Always inspect the `panel_*.png` QC panels before using the numbers.** Confirm that the mask
neither includes non-droplet material (red annotation marks, edge artefacts) nor misses faintly
stained droplets.

## Comparability notes

- A 10-unit change in the threshold shifts the absolute area fraction by roughly 2–3 percentage
  points. Between-group rankings and fold differences are stable; absolute values should be
  compared only within a batch.
- Absolute values from different batches (different sample types, staining intensity or exposure)
  are not directly comparable.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the archived release. The DOI is listed in `CITATION.cff`.
