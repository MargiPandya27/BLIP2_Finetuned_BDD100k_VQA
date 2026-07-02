# Dummy BDD100K Dataset

A tiny local substitute for the full BDD100K download (~5 samples, small JPEGs).

## Layout

```
dummy_bdd100k/
├── bdd100k/
│   ├── labels/bdd100k_labels_images_val.json
│   └── images/100k/val/*.jpg
└── processed/bdd100k_risk_vqa.json
```

## Regenerate

```bash
python scripts/generate_dummy_bdd100k.py
```

## Use in training

Point your data paths to:

- **Labels:** `data/dummy_bdd100k/bdd100k/labels/bdd100k_labels_images_val.json`
- **Images:** `data/dummy_bdd100k/bdd100k/images/100k/val/`
- **VQA (ready to train):** `data/dummy_bdd100k/processed/bdd100k_risk_vqa.json`

The VQA file uses absolute image paths so the dataloader can find files from any working directory.

## Scenarios included

| Sample | Scenario |
|--------|----------|
| `dummy_001_green_clear.jpg` | Green light, far vehicle |
| `dummy_002_red_close_car.jpg` | Red light, close vehicle |
| `dummy_003_yellow_medium_car.jpg` | Yellow light, medium-distance vehicle |
| `dummy_004_no_hazards.jpg` | No traffic signal or centered vehicle |
| `dummy_005_close_car_only.jpg` | Close vehicle, no traffic light |
