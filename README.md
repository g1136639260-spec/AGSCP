# AGSCP Demo

本仓库当前主要提供一个推理 demo，核心脚本是 `AG_person_search/search-det-seg-AG.py`。这个脚本会把空中检测、query 构建、地面 person search、分割可视化四个阶段串起来，最终输出带检测结果的视频、用于 ReID 的 query 图像、person search 的匹配结果，以及分割结果视频。

## 1. 运行前说明

- 建议使用 Python 3.10。
- 建议使用带 CUDA 的环境。脚本里的 MMDetection 和 MMSegmentation 初始化默认使用 `cuda:0`，当前没有整理出纯 CPU 运行路径。
- 第二阶段需要 OpenCV 图形界面交互。也就是说，运行环境必须能弹出窗口，不能是完全 headless 的终端。
- 仓库中的大体积 `.pth` 权重文件不适合直接提交到普通 GitHub 仓库。如果你的匿名仓库里不放权重，请在本地准备好对应 checkpoint，并通过命令行参数传入。

## 2. 安装环境

先安装与你机器匹配的 PyTorch 和 torchvision。CUDA 版本请按你自己的驱动环境选择。

然后安装 OpenMMLab 1.x 兼容栈：

```bash
pip install openmim
mim install "mmcv-full>=1.7,<1.8"
mim install "mmdet>=2.28,<2.29"
mim install "mmsegmentation>=0.30,<0.31"
pip install -r requirements-demo.txt
```

这套版本是按当前代码里的 API 用法整理出来的最小 demo 依赖，而不是简单照搬训练服务器上的整份 `pip list`。根目录下原有的 `requirements.txt` 仍然保留为服务器环境快照，真正用于安装的是 `requirements-demo.txt`。

## 3. Demo 流程

`AG_person_search/search-det-seg-AG.py` 的执行流程如下：

1. 空中检测：读取空中视角视频或图像，输出带检测框的可视化结果。
2. 构建 query：从指定视频中手动框选目标，保存成 ReID query 图像。
3. 地面 person search：在地面视角视频中搜索与 query 最匹配的目标，输出匹配结果视频和待分割 crop。
4. 分割与合成视频：对第三阶段导出的 crop 做分割，并把结果重新合成为视频。

## 4. 运行方法

在项目根目录执行。下面给的是参数模板，不包含任何真实机器路径。

PowerShell 可以直接这样写；如果你在 Linux 或 macOS 下运行，把每行结尾的反引号改成反斜杠即可。

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

### 可选参数

- `--query_input`
  - 可选。
  - 留空时默认使用 `--A_D_output` 作为 query 框选输入。
  - 如果你更希望直接从原始空中视频框选 query，就显式传入原视频路径。

- `--data`
  - 可选。
  - 留空时默认使用仓库自带的 `AG_person_search/data/AGR.data`。

- `--G_S_input_path`
  - 可选。
  - 留空时默认使用 `--gs_output`，也就是第三阶段导出的 crop 目录。

- `--find_class`
  - 默认值是 `person`。
  - 当前仓库的类别名定义在 `AG_person_search/data/AGR.names`，可选值为 `car`、`person`、`inf`。

## 5. 第二阶段的交互方式

程序进入 query 构建阶段后会弹出一个 OpenCV 窗口：

1. 点击窗口里的 `Start` 按钮。
2. 在画面中用鼠标拖拽框选目标。
3. 每框选一次，就会在 `--query_output` 目录下保存一张 query 图像。
4. 按 `Space` 切到下一帧。
5. 按 `Enter` 结束标注并进入 person search。
6. 按 `q` 可提前退出。

注意：

- 不要随意改脚本自动保存出来的 query 文件名格式，ReID 数据读取代码依赖类似 Market1501 的命名规则。
- 如果你打算自己提前准备 query 图像，也要保持兼容的命名格式。

## 6. 输出结果对应关系

- `--A_D_output`
  - 第一阶段的空中检测可视化结果。

- `--query_output`
  - 第二阶段手工框选得到的 query 图像目录。

- `--ps_output`
  - 第三阶段输出的 person search 结果视频目录。

- `--gs_output`
  - 第三阶段导出的待分割 crop 目录。

- `--G_S_output_path`
  - 第四阶段输出的分割结果图像目录。

- `--G_S_video_output_path`
  - 第四阶段把分割结果重新拼成的视频文件。

## 7. Demo 所需文件

运行这个 demo 至少需要准备：

- 空中检测模型的 config 和 checkpoint。
- 地面检测模型的 config 和 checkpoint。
- 分割模型的 config 和 checkpoint。
- ReID checkpoint。
- 一段空中视角输入视频或图像。
- 一段地面视角输入视频或图像。

仓库中的 `weights/*.py` 可以作为 OpenMMLab 配置示例。若匿名仓库不提供 `.pth` 文件，请使用你本地保存的 checkpoint 路径来运行。

## 8. 常见问题

### 窗口没有弹出

说明当前环境不支持 OpenCV GUI。这个脚本的第二阶段依赖手动框选，必须切换到有桌面的环境，或者自行改造成非交互式 query 输入。

### 程序一开始就报找不到路径

现在脚本里的路径默认值都只是占位说明。运行前需要把命令中的占位符全部替换成你自己的真实路径。

### 推到 GitHub 时报大文件错误

普通 GitHub 仓库不能直接接收超过 100 MB 的单文件。这个项目里的部分 `.pth` 超过了这个限制，公开代码仓库建议只提交代码和配置，把 checkpoint 放到发行页、网盘或 Git LFS。
