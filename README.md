# AGSCP Demo

This repository currently provides an inference demo. The main entry script is `AG_person_search/search-det-seg-AG.py`. The script connects four stages into one pipeline: aerial detection, query construction, ground-view person search, and segmentation visualization. The final outputs include a detection video, ReID query images, person-search results, and a segmentation video.

## 1. Before You Run

- Python 3.10 is recommended.
- A CUDA-enabled environment is recommended. The current MMDetection and MMSegmentation initialization paths in the script use `cuda:0`, and a CPU-only execution path has not been cleaned up yet.
- The second stage requires OpenCV GUI interaction. In practice, this means the environment must be able to open a display window and cannot be fully headless.
- Large `.pth` weight files are not suitable for direct upload to a standard GitHub repository. If your anonymous repository does not include weights, prepare the required checkpoints locally and pass them through command-line arguments.

## 2. Environment Setup

First install the PyTorch and torchvision versions that match your machine and CUDA runtime.

Then install the OpenMMLab 1.x-compatible stack:

```bash
pip install openmim
mim install "mmcv-full>=1.7,<1.8"
mim install "mmdet>=2.28,<2.29"
mim install "mmsegmentation>=0.30,<0.31"
pip install -r requirements-demo.txt
```

These versions were selected based on the API usage in the current codebase. They are intended as a minimal installable demo environment, rather than a direct copy of the full `pip list` from the original training server. The original `requirements.txt` is still kept in the root directory as a server environment snapshot. The actual installable dependency file is `requirements-demo.txt`.

## 3. Demo Pipeline

`AG_person_search/search-det-seg-AG.py` runs the following stages:

1. Aerial detection: read an aerial-view video or image and save the detection visualization result.
2. Query construction: manually crop the target from a selected video and save the crop or crops as ReID query images.
3. Ground-view person search: search for the target in a ground-view video and save both the matching result video and the crops for segmentation.
4. Segmentation and video assembly: run segmentation on the exported crops and merge the segmentation outputs back into a video.

## 4. How to Run

Run the script from the project root directory. The example below is only a parameter template and does not include any real machine-specific paths.

In PowerShell, you can write the command exactly as shown below. On Linux or macOS, replace the trailing backticks with backslashes.

```powershell
python AG_person_search/search-det-seg-AG.py `
  --A_D_config <AERIAL_DET_CONFIG> `
  --A_D_checkpoint <AERIAL_DET_CKPT> `
  --A_D_input <AERIAL_INPUT_VIDEO_OR_IMAGE> `
  --A_D_output <AERIAL_DET_OUTPUT_VIDEO> `
  --query_output <QUERY_OUTPUT_DIR> `
  --reid_weight <REID_CKPT> `
  --cfg <GROUND_DET_CONFIG> `
  --weights <GROUND_DET_CKPT> `
  --images <GROUND_INPUT_VIDEO_OR_IMAGE> `
  --ps_output <PERSON_SEARCH_OUTPUT_DIR> `
  --gs_output <PERSON_SEARCH_CROP_DIR> `
  --G_S_config_file <SEG_CONFIG> `
  --G_S_checkpoint_file <SEG_CKPT> `
  --G_S_output_path <SEG_MASK_OUTPUT_DIR> `
  --G_S_video_output_path <SEG_VIDEO_OUTPUT>
```

### Optional Arguments

- `--query_input`
  - Optional.
  - If left blank, the script uses `--A_D_output` as the input for interactive query cropping.
  - If you prefer to crop the query directly from the original aerial video, pass that video path explicitly.

- `--data`
  - Optional.
  - If left blank, the script uses the bundled `AG_person_search/data/AGR.data`.

- `--G_S_input_path`
  - Optional.
  - If left blank, the script uses `--gs_output`, which is the crop directory exported by the person-search stage.

- `--find_class`
  - The default value is `person`.
  - The current class names are defined in `AG_person_search/data/AGR.names`, with available values `car`, `person`, and `inf`.

## 5. Interactive Query Annotation

When the script enters the query construction stage, it opens an OpenCV window:

1. Click the `Start` button in the window.
2. Drag the mouse to draw a box around the target.
3. Each crop is saved into `--query_output`.
4. Press `Space` to move to the next frame.
5. Press `Enter` to finish annotation and continue to person search.
6. Press `q` to exit early.

Notes:

- Do not arbitrarily change the auto-generated query filename format. The ReID data loader expects a Market1501-like naming convention.
- If you prepare query images manually, keep the filenames in a compatible format.

## 6. Output Mapping

- `--A_D_output`
  - Output of the aerial detection visualization stage.

- `--query_output`
  - Directory of query images created during manual annotation.

- `--ps_output`
  - Output directory for the person-search result video.

- `--gs_output`
  - Directory of crops exported after person search and used as segmentation input.

- `--G_S_output_path`
  - Directory of segmentation result images.

- `--G_S_video_output_path`
  - Video assembled from the segmentation outputs.

## 7. Required Demo Files

At minimum, the demo requires:

- An aerial detection config and checkpoint.
- A ground-view detection config and checkpoint.
- A segmentation config and checkpoint.
- A ReID checkpoint.
- One aerial-view input video or image.
- One ground-view input video or image.

The `weights/*.py` files included in this repository can be used as OpenMMLab config examples. If the anonymous repository does not include `.pth` files, use the local checkpoint paths on your own machine when running the demo.
