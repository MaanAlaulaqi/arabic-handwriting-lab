# Exact set0 manifests

These portable copies preserve the original 22,092 training, 1,069 validation, and 1,334 test line assignments, text, source-page IDs, and image/label SHA-256 fields. Only the `image_path` and `label_path` values changed from machine-specific absolute paths to `public/<filename>`. The workflow resolves each line by filename inside the official `public_line_images.zip` archive.

The original manifest verification reported 1,100 / 50 / 66 source pages with no page overlap and complete coverage of 24,495 public lines. The supplied test manifest's image names and exact `text` values were independently compared with both published test prediction files.

Source dataset: [Muharaf-public](https://doi.org/10.5281/zenodo.11492215), Saeed and collaborators. The image archive is downloaded from the source record; it is not stored in this repository.
