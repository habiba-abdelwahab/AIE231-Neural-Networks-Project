# Data: HAM10000

**HAM10000** ("Human Against Machine with 10000 training images"): 10,015 dermatoscopic
images of pigmented skin lesions in 7 diagnostic classes, with metadata
(lesion_id, image_id, dx, dx_type, age, sex, localization).

- Source: Tschandl, Rosendahl & Kittler (2018), *Scientific Data* 5:180161, doi:10.1038/sdata.2018.161.
  Harvard Dataverse doi:10.7910/DVN/DBW86T. Mirror used by `download.py`: Kaggle `kmader/skin-cancer-mnist-ham10000`.
- Licence: CC BY-NC 4.0 (academic, non-commercial use allowed; cite the paper).
- Download: `python data/download.py` puts everything in `data/raw/ham10000/` (about 3 GB, not committed).

| Code | Diagnosis |
|---|---|
| akiec | Actinic keratoses / intraepithelial carcinoma (pre-cancer) |
| bcc | Basal cell carcinoma (cancer) |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma (benign) |
| mel | Melanoma (dangerous cancer) |
| nv | Melanocytic nevi, i.e. moles (benign) |
| vasc | Vascular lesions (benign) |
