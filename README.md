# EVOD-RoI preprocessing

This repository contains only the small preprocessing utility used to create
the public EVOD-RoI data attachment. The generated images and labels are kept
outside GitHub because they are too large for a source-code repository.

## What the script does

`preprocess_bdd100k.py` follows the procedure used by the original anonymous
attachment:

1. read BDD frames and their YOLO (or YOLO-with-track-id) annotations;
2. find the half-frame window containing the most annotated object centres;
3. resize the complete frame to one-quarter resolution and resize it back; and
4. restore the selected half-frame window at its original resolution.

The output keeps the original image dimensions and writes ordinary five-field
YOLO labels. Track IDs, when present in the input, are omitted from the output
labels. The annotations are used only to select the clear region and to write
the matching labels; this script is an offline data-preparation utility, not
the online detector/ROI module described in the paper. It intentionally
reproduces the fixed one-quarter-resolution public visualization; it is not a
replacement for the paper's internal five-frame detector or four-ratio
training generator.

## Data

The expanded data attachment is distributed separately. Replace the two
placeholders below with the final cloud URLs before publishing this README:

- BDD100K processed attachment (8,000 train + 2,000 validation frames):
  `<BDD100K_CLOUD_URL>`
- Optional COCO visualization attachment (legacy demonstration only):
  `<COCO_CLOUD_URL>`

The cloud archive must include the applicable source-dataset license and
attribution. The images in the attachment are processed derivatives and should
not be presented as a redistribution of the original BDD100K or COCO source
dataset. The BDD source used during development was the vehicle-only mirror
listed at <https://huggingface.co/datasets/vanthanh/bdd100kmot_vehicle>; it is
not the complete official BDD100K tracking release. The COCO link is optional
and is not required to reproduce the BDD100K fine-tuning attachment.

## Run

Create an environment with Python 3.11, NumPy, and OpenCV (for example, with
Conda):

```bash
conda create -n evod-roi python=3.11 numpy opencv -c conda-forge
conda activate evod-roi
```

The BDD source directory must have this layout:

```text
source/
  images/{train,val}/<sequence>/img1/*.jpg
  labels_with_ids/{train,val}/<sequence>/img1/*.txt
```

Run the deterministic 10,000-frame expansion as follows:

```bash
python preprocess_bdd100k.py \
  --source-root /data/bdd100kmot_vehicle \
  --output-dir /data/evod_roi_bdd100k \
  --total 10000 \
  --train-count 8000 \
  --seed 42 \
  --labeled-preview-count 100
```

If legacy BDD metadata is available, add `--metadata-root` to keep daytime
`city street` and `highway` sequences only. `--license-file` and
`--attribution-file` can be used to copy the corresponding notices into the
generated data directory. The command refuses to overwrite a non-empty output
directory and writes `manifest.csv`, `summary.json`, `dataset.yaml`, and
`files.sha256` alongside the generated images and labels.

## Citation and attribution

Please cite the EVOD-RoI paper and the original BDD100K/COCO dataset papers
when using the attachment. Keep the original dataset terms with every cloud
copy.
