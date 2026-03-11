import argparse
import time
from models import *
from utils.utils import *
from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
# from configs.reid.config import cfg as reidCfg
from mmdet.apis import init_detector, inference_detector
global frame#后面查一下完整的代码是否需要这样设置
global point1, point2#后面查一下完整的代码是否需要这样设置
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import natsort
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)
# 获取当前文件所在目录（即 AG_person_search 文件夹）
current_dir = os.path.dirname(__file__)
# 回到上一级目录（即项目根目录 2030pj-finish-for-car）
ROOT = os.path.abspath(os.path.join(current_dir, '..'))
# 把根目录加入 Python 模块搜索路径
sys.path.append(ROOT)
from reid.config import cfg as reidCfg

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, '..'))



def G_person_search(cfg,
           data,
           weights,
           images,  # input folder
           Reid_output,
           GSI_output,  # output folder
           fourcc,  # video codec
           img_size,
           conf_thres,
           nms_thres,
           dist_thres,  # 距离阈值
           find_class='person',
           save_txt=False,
           save_images=True,):

    # Initialize
    device = torch_utils.select_device(force_cpu=False)  # 选择设备
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(GSI_output):
        shutil.rmtree(GSI_output)  # delete output folder
    os.makedirs(GSI_output)  # make new output folder

    ############# 行人重识别模型初始化 #############
    query_loader, num_query = make_data_loader(reidCfg)  # 验证集预处理
    # reidModel = build_model(reidCfg, num_classes=10126)
    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化,这个classes无所谓，设置多少都可以，后面也用不到
    reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
    reidModel.to(device).eval()  # 模型测试

    query_feats = []  # 测试特征
    query_pids = []  # 测试ID

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch  # 返回图片，ID，相机ID
            img = img.to(device)      # 将图片放入gpu
            feat = reidModel(img)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats.append(feat)       # 获得特征值列表
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量

    ############# 行人检测模型初始化-our #############
    device = 'cuda:0'
    model = init_detector(cfg, weights, device=device)


    ############## Set Dataloader################
    video = mmcv.VideoReader(images)
    # 获取视频参数
    fps = video.fps  # 获取视频的帧率
    width = video.width  # 获取视频宽度
    height = video.height  # 获取视频高度

    # 初始化一个列表，用于存储所有处理后的帧
    frames = []

    ############## get classes and colors################
    # 下面这两行代码功能就是Get classes and colors，但是这个classes和你的模型无关，且这真的只是单纯的导入一下，和coco格式也无关，只是借用了coco的api进行导入类别，这个classes的txt文件你写啥类别，这边就是啥类别
    # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
    # 从names=data/coco.names这里设置好的类别名称又通过下面这个代码提取出来了，colors也对应加上了
    classes = load_classes(parse_data_cfg(data)['names'])  # 得到类别名列表: ['person', 'bicycle'...]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 对于每种类别随机使用一种颜色画框

    ############## Run inference ################
    t0 = time.time()
    D = []
    for frame_idx, frame in enumerate(video):
        t = time.time()
        ############## 这里没有resize################
        # 推理模型得到结果
        det = inference_detector(model, frame)
        # 取出 'person' 类别的检测框
        find_class_idx = classes.index(find_class)  # 获取 'person' 类别索引
        find_class_bboxes = det[find_class_idx]  # 获取 'person' 类别的框
        ############## reid代码的推理及输出结果-这里只对person类别进行处理 ################
        if det is not None and len(det) > 0:
            # 如果没有检测到人，跳过
            if find_class_bboxes is None or len(find_class_bboxes) == 0:
                print(f'未检测到{find_class}')
                # continue

            # 过滤置信度低的框
            score_thr = 0
            find_class_bboxes = np.array(find_class_bboxes)
            keep = find_class_bboxes[:, 4] >= score_thr
            find_class_bboxes = find_class_bboxes[keep]
            if len(find_class_bboxes) == 0:
                print(f'person 全部低于阈值 {score_thr}')
                continue

            # 转换为 torch.Tensor 格式（符合后续处理要求）
            find_class_bboxes = torch.from_numpy(find_class_bboxes)

            # 没有resize所以这里也先不进行->>>映射回原图大小
            # find_class_bboxes[:, :4] = scale_coords(img.shape[2:], find_class_bboxes[:, :4], im0.shape).round()

            # 打印检测到的人物数量
            print(f'Detected {len(find_class_bboxes)} persons')

            # 将检测到的框进行裁剪并保存
            gallery_img = []
            seg_input_img = []
            gallery_loc = []  # 存放框的坐标
            for *xyxy, conf in find_class_bboxes:  # 遍历每个人物框
                xmin, ymin, xmax, ymax = map(int, xyxy)  # 获取框的坐标
                w = xmax - xmin  # 框的宽度
                h = ymax - ymin  # 框的高度

                # 过滤掉过小的框
                if w * h > 256:  # 可以根据需要调整这个阈值
                    gallery_loc.append((xmin, ymin, xmax, ymax))
                    crop_img = frame[ymin:ymax, xmin:xmax]  # 从原图中裁剪出人物图像
                    seg_input_img.append(crop_img)
                    # 保存到文件夹，文件名可以用 idx 或者时间戳
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # 转为 PIL 图像
                    crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # 转为 ReID 输入格式
                    gallery_img.append(crop_img)


            ############## 计算query和gallery的阈值并将gallery中视频的框画上输出结果 ################
            if gallery_img:
                gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                gallery_img = gallery_img.to(device)
                gallery_feats = reidModel(gallery_img) # torch.Size([7, 2048])
                print("The gallery feature is normalized")
                gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                m, n = query_feats.shape[0], gallery_feats.shape[0]
                distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                distmat = distmat.sum(axis=0) / len(query_feats) # 平均一下query中同一行人的多个结果
                index = distmat.argmin()
                if distmat[index] < dist_thres:
                    # 将裁剪信息保存到 D
                    D.append({
                        "frame_idx": frame_idx,
                        "class": find_class,
                        "bbox": gallery_loc[index]
                    })
                    ##################保存seg需要用的图片#####
                    # crop_file_name = f"crop_{frame_idx}_{find_class}_{index}.jpg"  # frame_idx 表示帧索引，idx 表示人物编号
                    # crop_file_path = os.path.join(GSI_output, crop_file_name)
                    # cv2.imwrite(crop_file_path, seg_input_img[index])  # 保存 BGR 图像

                    print('距离：%s'%distmat[index])
                    plot_one_box(gallery_loc[index], frame, label='find!', color=colors[int(find_class_idx)])

        print('Done. (%.3fs)' % (time.time() - t))
        frames.append(frame)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + Reid_output)
        file_name = os.path.basename(images)  # 获取文件名，包括扩展名
        file_name_without_ext, ext = os.path.splitext(file_name)  # 分离文件名和扩展名
        # 在文件名末尾添加 "_searched" 并保留扩展名
        new_file_name = file_name_without_ext + "_searched" + ext
        output_video_path = Reid_output+"/"+new_file_name # 保存的路径
        vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        # 将所有帧写入视频
        for frame in frames:
            vid_writer.write(frame)  # 将每帧写入视频文件
        # 释放资源
        vid_writer.release()
        print(f"Video saved to {output_video_path}")
        # cv2.destroyAllWindows()

    #截取分割所需要的图像
    # 先计算 H_max 和 W_max
    H_max, W_max = 0, 0
    for item in D:
        xmin, ymin, xmax, ymax = item["bbox"]
        w = xmax - xmin
        h = ymax - ymin
        W_max = max(W_max, w)
        H_max = max(H_max, h)

    print(f"H_max={H_max}, W_max={W_max}")

    # 遍历视频帧，根据 D 中信息截取统一尺寸图像
    seg_output_dir = GSI_output  # 或者指定一个新的保存目录
    os.makedirs(seg_output_dir, exist_ok=True)

    for item in D:
        frame_idx = item["frame_idx"]
        xmin, ymin, xmax, ymax = item["bbox"]
        class_name = item["class"]

        frame = video[frame_idx]  # 获取对应帧
        H, W, C = frame.shape

        # 计算 bbox 中心点
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        # 计算截取区域坐标
        x1 = cx - W_max // 2
        y1 = cy - H_max // 2
        x2 = x1 + W_max
        y2 = y1 + H_max

        # 判断是否超出图像边界
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            print(f"Frame {frame_idx} bbox {xmin, ymin, xmax, ymax} 超出边界，已删除")
            continue

        # 截取图像
        crop_img = frame[y1:y2, x1:x2]

        # 保存
        crop_file_name = f"crop_{frame_idx}_{class_name}.jpg"
        crop_file_path = os.path.join(seg_output_dir, crop_file_name)
        cv2.imwrite(crop_file_path, crop_img)

    print('Done. (%.3fs)' % (time.time() - t0))





def A_detect_and_save(A_D_config_file, A_D_checkpoint_file, A_D_input_path, A_D_output_path):
    """
    通过MMDet模型进行空中数据源的检测并保存结果（支持图像和视频）。
    - 若输入为图像：A_D_output_path 为输出文件路径（如 /path/out.jpg）
    - 若输入为视频：A_D_output_path 可为“输出视频文件路径”（如 /path/out.mp4）
                    或“输出目录”（如 /path/out_dir/，将生成 inputname_vis.mp4）
    """
    # 1) 构建模型（默认使用0号GPU）
    model = init_detector(A_D_config_file, A_D_checkpoint_file, device='cuda:0')

    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.wmv')
    is_video = A_D_input_path.lower().endswith(video_exts)

    if is_video:
        # —— 视频：边推理边写视频文件（OpenCV VideoWriter） ——
        video = mmcv.VideoReader(A_D_input_path)

        # 决定输出视频路径：既支持传入文件，也支持传入目录
        if A_D_output_path.lower().endswith(video_exts):
            out_video_path = A_D_output_path
            out_dir = os.path.dirname(out_video_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        else:
            os.makedirs(A_D_output_path, exist_ok=True)
            base = os.path.splitext(os.path.basename(A_D_input_path))[0]
            out_video_path = os.path.join(A_D_output_path, f"{base}_vis.mp4")

        # fps/size 兼容：若读不到 fps，就用 25；尺寸用首帧来确定更稳
        fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 常用；若不兼容可改 'avc1' 或 'XVID'
        fps = getattr(video, 'fps', None) or 25

        writer = None  # 延后到拿到第一帧可视化图时再初始化（避免宽高未知）

        for idx, frame in enumerate(video):
            result = inference_detector(model, frame)

            # —— 获取可视化帧（兼容不同 MMDet 版本） ——
            vis_frame = None
            try:
                vis_frame = model.show_result(frame, result, show=False, out_file=None)
            except TypeError:
                try:
                    vis_frame = model.show_result(frame, result, show=False)
                except Exception:
                    vis_frame = None

            # 兜底：有些版本不返回 ndarray，就先落地再读回
            if vis_frame is None:
                tmp_dir = os.path.dirname(out_video_path) or '.'
                tmp_path = os.path.join(tmp_dir, '__tmp_vis.jpg')
                model.show_result(frame, result, show=False, out_file=tmp_path)
                vis_frame = mmcv.imread(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            # 首帧时初始化 VideoWriter（用首帧的尺寸）
            if writer is None:
                h, w = vis_frame.shape[:2]
                writer = cv2.VideoWriter(out_video_path, fourcc_code, fps, (w, h))

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open VideoWriter for {out_video_path}. "
                        f"Try changing codec to 'avc1' (H.264) or确保本机已安装对应编码器。"
                    )

            writer.write(vis_frame)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} frames...")

        if writer is not None:
            writer.release()
        print(f"Processed video and saved to {out_video_path}")

    else:
        # —— 图像：直接把可视化结果写入到 A_D_output_path 文件 ——
        img = mmcv.imread(A_D_input_path)
        result = inference_detector(model, img)

        out_dir = os.path.dirname(A_D_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        model.show_result(img, result, show=False, out_file=A_D_output_path)
        print(f"Processed image and saved to {A_D_output_path}")


def G_Seg_and_save(G_S_config_file, G_S_checkpoint_file, G_S_input_path, G_S_output_path):
    """
    支持对文件夹下所有图像进行分割并保存结果。
    """
    # 建立模型
    model = init_segmentor(G_S_config_file, G_S_checkpoint_file, device='cuda:0')

    # 确保输出目录存在
    os.makedirs(G_S_output_path, exist_ok=True)

    # 如果输入是目录，遍历目录下所有图片
    if os.path.isdir(G_S_input_path):
        img_files = [os.path.join(G_S_input_path, f) for f in os.listdir(G_S_input_path)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        img_files = [G_S_input_path]  # 单张图像

    for img_path in img_files:
        print(f"Processing {img_path} ...")
        img = mmcv.imread(img_path)
        result = inference_segmentor(model, img)

        # 构建保存路径，保持原图名
        save_path = os.path.join(G_S_output_path, os.path.basename(img_path))
        show_result_pyplot(model, img, result, out_file=save_path)
        print(f"Saved result to {save_path}")




# 在程序初始化时，设置窗口属性
cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # 创建可缩放窗口
cv2.resizeWindow("image", 1000, 700)          # 指定窗口大小



def _draw_intro(img, st):
    """在 img 上居中叠加中英文提示 + Start 按钮（PIL 渲染中文）"""
    disp = img.copy()
    H, W = disp.shape[:2]

    # 1) 中文字体路径：优先用 state['font_path']，否则探测常见路径
    font_path = st.get('font_path') if isinstance(st, dict) else None
    if not font_path:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/Supplemental/Heiti TC.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break
    if not font_path:
        print("[Warn] Chinese-capable font not found. Set state['font_path'] to a valid .ttf/.ttc (e.g., NotoSansCJK).")

    # 2) PIL 画布
    pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # 3) 字体（你可以自由改 size / stroke 宽度以调粗）
    f_title = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=40)
    f_head  = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=38)
    f_body  = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=30)
    stroke  = 3

    # 4) 英中分行内容（英文一行，中文下一行）
    lines = [
        ("Start calibrating the target that the self-driving car needs to track",
         "开始标定无人车需要跟踪的目标"),
        ("Operation:", "操作说明："),
        ("1. Select the target location and drag the mouse to frame the target to be tracked",
         "1. 选择目标位置并拖动鼠标框选需要跟踪的目标"),
        ("2. Press Enter after the frame is set",
         "2. 框选完成后按回车键"),
    ]
    # 每个 pair（英文,中文）对应的字体/颜色
    fonts_en  = [f_title, f_head, f_body, f_body]
    fonts_cn  = [f_title, f_head, f_body, f_body]
    colors_en = [(255,210,0), (0,220,0), (255,255,255), (255,255,255)]
    colors_cn = [(255,210,0), (0,220,0), (255,255,255), (255,255,255)]

    gap_between_lines = 8     # 英文与其中文之间的间距
    gap_between_pairs = 16    # 每个(英文+中文)块之间的间距

    # 5) 先量总高度用于垂直居中
    pair_heights = []
    for (en, cn), fe, fc in zip(lines, fonts_en, fonts_cn):
        be = draw.textbbox((0, 0), en, font=fe, stroke_width=stroke)
        bc = draw.textbbox((0, 0), cn, font=fc, stroke_width=stroke)
        he = be[3] - be[1]
        hc = bc[3] - bc[1]
        pair_heights.append(he + gap_between_lines + hc)
    total_h = sum(pair_heights) + gap_between_pairs * (len(lines) - 1)

    # 6) 逐 pair 居中绘制：英文一行 + 中文下一行
    y = (H - total_h) // 2
    for (en, cn), fe, fc, ce, cc, ph in zip(lines, fonts_en, fonts_cn, colors_en, colors_cn, pair_heights):
        be = draw.textbbox((0, 0), en, font=fe, stroke_width=stroke)
        we = be[2] - be[0]
        xe = (W - we) // 2
        draw.text((xe, y), en, font=fe, fill=ce, stroke_width=stroke, stroke_fill=(0,0,0))

        bc = draw.textbbox((0, 0), cn, font=fc, stroke_width=stroke)
        wc = bc[2] - bc[0]
        hc = bc[3] - bc[1]
        xc = (W - wc) // 2
        y_cn = y + (be[3]-be[1]) + gap_between_lines
        draw.text((xc, y_cn), cn, font=fc, fill=cc, stroke_width=stroke, stroke_fill=(0,0,0))

        y += ph + gap_between_pairs

    # 7) 转回 BGR，画 Start 按钮
    disp = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bw, bh = 200, 56
    x1 = (W - bw) // 2
    y1 = y + 24
    x2, y2 = x1 + bw, y1 + bh
    if isinstance(st, dict):
        st['button_rect'] = (x1, y1, x2, y2)

    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 180, 255), -1)
    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 100, 200), 2)

    # 按钮文字（仍用 PIL 居中）
    pil2 = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(pil2)
    f_btn = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=28)
    btn_text = "Start"
    bb = draw2.textbbox((0, 0), btn_text, font=f_btn, stroke_width=2)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    tx = x1 + (bw - tw) // 2
    ty = y1 + (bh - th) // 2
    draw2.text((tx, ty), btn_text, font=f_btn, fill=(20,20,20), stroke_width=2, stroke_fill=(255,255,255))

    return cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)


def _in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def query_get(event, x, y, flags, param):
    """OpenCV 鼠标回调：从 param 中读取当前帧与保存目录"""
    state = param  # dict: {'frame': ..., 'p1': ..., 'save_dir': ..., 'count': ..., 'prefix': ...}
    frame = state.get('frame', None)
    if frame is None:
        return

    # —— 给现有 state “按需补充”两个键，不会依赖外部全局 frame ——
    if 'started' not in state:
        state['started'] = False
    if 'button_rect' not in state:
        state['button_rect'] = (50, 180, 180, 220)  # (x1,y1,x2,y2) 可按需改

    # —— 未开始：只显示引导界面；点击“Start”后进入标定 ——
    if not state['started']:
        if event == cv2.EVENT_LBUTTONDOWN and _in_rect(x, y, state['button_rect']):
            state['started'] = True
            cv2.imshow('image', frame)  # 切回纯画面，进入标定
        else:
            cv2.imshow('image', _draw_intro(frame, state))
        return

    # —— 已开始：沿用你原有的标定/裁剪保存逻辑 ——
    img2 = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        state['p1'] = (x, y)
        cv2.circle(img2, state['p1'], 10, (0, 255, 0), 3)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and state.get('p1') is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, state['p1'], (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP and state.get('p1') is not None:
        p1 = state['p1']
        p2 = (x, y)
        min_x, min_y = min(p1[0], p2[0]), min(p1[1], p2[1])
        width, height = abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

        # 可视化最终框
        cv2.rectangle(img2, p1, p2, (0, 0, 255), 2)
        cv2.imshow('image', img2)

        if width > 0 and height > 0:
            cut_img = frame[min_y:min_y+height, min_x:min_x+width]
            os.makedirs(state['save_dir'], exist_ok=True)
            fname = f"{state['prefix']}_{state['count']:04d}_x{min_x}_y{min_y}.jpg"
            save_path = os.path.join(state['save_dir'], fname)
            cv2.imwrite(save_path, cut_img)
            print(f"[Saved] {save_path}")
            state['count'] += 1

        # 清空起点
        state['p1'] = None




def letterbox_image(image, target_size, color=(0, 0, 0)):
    """保持比例缩放并用颜色填充"""
    ih, iw = image.shape[:2]
    h, w = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh))
    new_image = np.full((h, w, 3), color, dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    new_image[top:top+nh, left:left+nw] = resized
    return new_image


def images_to_video(input_folder, output_video_path, fps=25):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"❌ 输入文件夹不存在: {input_folder}")

    if os.path.isdir(output_video_path) or not os.path.splitext(output_video_path)[1]:
        output_video_path = os.path.join(output_video_path, "output.mp4")
        print(f"⚠️ 检测到输出路径是文件夹，已自动更正为: {output_video_path}")

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError(f"⚠️ 文件夹中未找到图片: {input_folder}")

    images = natsort.natsorted(images)
    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    if first_frame is None:
        raise ValueError("⚠️ 无法读取第一张图像")
    height, width, _ = first_frame.shape

    output_dir = os.path.dirname(output_video_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"❌ 无法创建视频文件: {output_video_path}")

    valid_frames = 0
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ 跳过无法读取的图像: {img_name}")
            continue
        frame = letterbox_image(frame, (height, width))  # ✅ 等比缩放 + 填充
        video_writer.write(frame)
        valid_frames += 1
        print(f"✅ 写入帧: {img_name}")

    video_writer.release()
    cv2.destroyAllWindows()

    if valid_frames == 0:
        print("⚠️ 没有有效帧被写入，视频为空！")
    else:
        print(f"✅ 视频已成功保存到: {os.path.abspath(output_video_path)} （共 {valid_frames} 帧）")

    return output_video_path


def _is_placeholder_path(path_value):
    if path_value is None:
        return True
    normalized = str(path_value).strip().replace('\\', '/')
    if not normalized:
        return True
    return normalized.startswith(('path/to/', '<', '[set'))


def _resolve_cli_path(path_value, fallback=None):
    if _is_placeholder_path(path_value):
        if fallback is None:
            return None
        path_value = fallback
    return os.path.abspath(os.path.expanduser(path_value))


def _require_cli_path(path_value, arg_name, fallback=None):
    resolved = _resolve_cli_path(path_value, fallback=fallback)
    if resolved is None:
        raise ValueError(
            f"Please provide a real path for --{arg_name}. "
            f"The current value is only a placeholder."
        )
    return resolved


def _configure_reid_runtime(query_dir, reid_weight):
    reidCfg.DATASETS.ROOT_DIR = query_dir
    reidCfg.TEST.WEIGHT = reid_weight


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--A_D_config',
        type=str,
        default='path/to/aerial_detection_config.py',
        help='Config file used by the aerial detector.',
    )
    parser.add_argument(
        '--A_D_checkpoint',
        type=str,
        default='path/to/aerial_detection_checkpoint.pth',
        help='Checkpoint file used by the aerial detector.',
    )
    parser.add_argument(
        '--A_D_input',
        type=str,
        default='path/to/aerial_input_video_or_image',
        help='Input video or image for aerial detection.',
    )
    parser.add_argument(
        '--A_D_output',
        type=str,
        default='path/to/aerial_detection_output.mp4',
        help='Output path for the aerial detection result.',
    )

    parser.add_argument(
        '--query_input',
        type=str,
        default='',
        help='Optional input for query cropping. Blank means using --A_D_output.',
    )
    parser.add_argument(
        '--query_output',
        type=str,
        default='path/to/query_output_dir',
        help='Directory used to save cropped query images.',
    )
    parser.add_argument(
        '--reid_weight',
        type=str,
        default='path/to/reid_checkpoint.pth',
        help='Checkpoint file used by the ReID model.',
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='path/to/ground_detection_config.py',
        help='Config file used by the ground-view detector.',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='',
        help='Optional class config file. Blank means using the bundled AGR.data.',
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='path/to/ground_detection_checkpoint.pth',
        help='Checkpoint file used by the ground-view detector.',
    )
    parser.add_argument(
        '--images',
        type=str,
        default='path/to/ground_input_video_or_image',
        help='Ground-view video or image used for person search.',
    )
    parser.add_argument(
        '--ps_output',
        type=str,
        default='path/to/person_search_output_dir',
        help='Directory used to save the person-search visualization.',
    )
    parser.add_argument(
        '--gs_output',
        type=str,
        default='path/to/person_search_crop_output_dir',
        help='Directory used to save the crops exported after person search.',
    )

    parser.add_argument('--img-size', type=int, default=416, help='Input resolution for the legacy detector utilities.')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS threshold.')
    parser.add_argument('--dist_thres', type=float, default=1.0, help='ReID distance threshold.')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='Video codec for output videos.')
    parser.add_argument('--half', action='store_true', help='Reserved flag. Kept for compatibility.')
    parser.add_argument('--webcam', action='store_true', help='Reserved flag. Kept for compatibility.')
    parser.add_argument('--find_class', type=str, default='person', help='Target class name used by person search.')

    parser.add_argument(
        '--G_S_config_file',
        type=str,
        default='path/to/segmentation_config.py',
        help='Config file used by the segmentation model.',
    )
    parser.add_argument(
        '--G_S_checkpoint_file',
        type=str,
        default='path/to/segmentation_checkpoint.pth',
        help='Checkpoint file used by the segmentation model.',
    )
    parser.add_argument(
        '--G_S_input_path',
        type=str,
        default='',
        help='Optional segmentation input directory. Blank means using --gs_output.',
    )
    parser.add_argument(
        '--G_S_output_path',
        type=str,
        default='path/to/segmentation_mask_output_dir',
        help='Directory used to save segmentation masks.',
    )
    parser.add_argument(
        '--G_S_video_output_path',
        type=str,
        default='path/to/segmentation_video_output.mp4',
        help='Output video path assembled from segmentation masks.',
    )

    opt = parser.parse_args()

    default_data_cfg = os.path.join(current_dir, 'data', 'AGR.data')

    opt.A_D_config = _require_cli_path(opt.A_D_config, 'A_D_config')
    opt.A_D_checkpoint = _require_cli_path(opt.A_D_checkpoint, 'A_D_checkpoint')
    opt.A_D_input = _require_cli_path(opt.A_D_input, 'A_D_input')
    opt.A_D_output = _require_cli_path(opt.A_D_output, 'A_D_output')
    opt.query_input = _resolve_cli_path(opt.query_input, fallback=opt.A_D_output)
    opt.query_output = _require_cli_path(opt.query_output, 'query_output')
    opt.reid_weight = _require_cli_path(opt.reid_weight, 'reid_weight')
    opt.cfg = _require_cli_path(opt.cfg, 'cfg')
    opt.data = _require_cli_path(opt.data, 'data', fallback=default_data_cfg)
    opt.weights = _require_cli_path(opt.weights, 'weights')
    opt.images = _require_cli_path(opt.images, 'images')
    opt.ps_output = _require_cli_path(opt.ps_output, 'ps_output')
    opt.gs_output = _require_cli_path(opt.gs_output, 'gs_output')
    opt.G_S_config_file = _require_cli_path(opt.G_S_config_file, 'G_S_config_file')
    opt.G_S_checkpoint_file = _require_cli_path(opt.G_S_checkpoint_file, 'G_S_checkpoint_file')
    opt.G_S_input_path = _resolve_cli_path(opt.G_S_input_path, fallback=opt.gs_output)
    opt.G_S_output_path = _require_cli_path(opt.G_S_output_path, 'G_S_output_path')
    opt.G_S_video_output_path = _require_cli_path(opt.G_S_video_output_path, 'G_S_video_output_path')

    _configure_reid_runtime(opt.query_output, opt.reid_weight)

    print(opt)

    A_detect_and_save(
        A_D_config_file=opt.A_D_config,
        A_D_input_path=opt.A_D_input,
        A_D_output_path=opt.A_D_output,
        A_D_checkpoint_file=opt.A_D_checkpoint
    )

    cap = cv2.VideoCapture(opt.query_input)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open query_input: {opt.query_input}")

    state = {
        'frame': None,
        'p1': None,
        'save_dir': opt.query_output,
        'count': 0,
        'prefix': '0001_c1s1_50_0'
    }
    os.makedirs(state['save_dir'], exist_ok=True)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', query_get, state)

    while True:
        success, frame = cap.read()
        if not success:
            break
        state['frame'] = frame
        cv2.imshow('image', frame)

        key = cv2.waitKey(0)
        if key == 13:
            print("Query annotation finished")
            break
        if key == 32:
            continue
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with torch.no_grad():
        G_person_search(
            opt.cfg,
            opt.data,
            opt.weights,
            images=opt.images,
            Reid_output=opt.ps_output,
            GSI_output=opt.gs_output,
            fourcc=opt.fourcc,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            dist_thres=opt.dist_thres,
            find_class=opt.find_class
        )

    G_Seg_and_save(
        opt.G_S_config_file,
        opt.G_S_checkpoint_file,
        opt.G_S_input_path,
        opt.G_S_output_path
    )

    images_to_video(opt.G_S_output_path, opt.G_S_video_output_path)


if __name__ == '__main__':
    main()


if False and __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ==== 空中检测 ====
    parser.add_argument('--A_D_config', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/A-D-weights/central_r50_fpn_Up-G-D-part123hut_routerv1-A.py',
                        help="空中实时检测-配置文件路径")
    parser.add_argument('--A_D_checkpoint', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/A-D-weights/epoch_12.pth',
                        help="空中实时检测-权重文件路径")
    parser.add_argument('--A_D_input', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/1A-D-I/c1s1_800.MP4',
                        help="空中实时检测-输入图像或视频文件路径")
    parser.add_argument('--A_D_output', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/2A-D-O_and_get-query-input/c1s1_800_detected.mp4',
                        help="空中实时检测-输出结果视频文件路径（建议给到文件而不是目录）")

    # ==== 构建 query 的输入/输出 ====
    parser.add_argument('--query_input', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/1A-D-I/c1s1_800.MP4',
                        help="构建query的input路径（视频或图像）")
    parser.add_argument('--query_output', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/3query_out',
                        help="生成query的output路径（文件夹）")

    # ==== person-search ====
    parser.add_argument('--cfg', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/G-D-weights/central_r50_fpn_Up-G-D-part123hut_routerv1-G.py',
                        help="person-search中检测模型配置文件路径")
    parser.add_argument('--data', type=str, default='data/AGR.data', help="数据集配置文件所在路径")
    parser.add_argument('--weights', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/G-D-weights/epoch_8.pth',
                        help='person-search中检测模型权重文件路径')
    parser.add_argument('--images', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/4Reid-gallery_and_G-D-I/c2s1_100.MP4',
                        help='地面需要进行检测并reid的图片或视频文件夹')
    parser.add_argument('--ps_output', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/5Person-search-Out',
                        help='person-search检测后的图片或视频保存的路径')
    parser.add_argument('--gs_output', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data-demo8/6G-S-I_and_proc-of-search-out',
                        help='person-search后处理结果保存路径')

    parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')
    parser.add_argument('--dist_thres', type=float, default=1.0, help='行人图片距离阈值')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='是否采用半精度FP16进行推理')
    parser.add_argument('--webcam', action='store_true', help='是否使用摄像头进行检测')
    parser.add_argument('--find_class', type=str, default='inf', help='所要寻找目标的类别-需标定以快速筛选')

    # ==== 分割 ====

    parser.add_argument('--G_S_config_file', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/G-S-weights/r50_UP-G_Seg-part123hut_routerv1—A.py',
                        help="Path to the config file.")
    parser.add_argument('--G_S_checkpoint_file', type=str,
                        default='/home/ubuntu/End2EndAG-github/weights/G-S-weights/iter_20000.pth',
                        help="Path to the checkpoint file.")
    parser.add_argument('--G_S_input_path', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/6G-S-I_and_proc-of-search-out',
                        help="Path to the input image file.")
    parser.add_argument('--G_S_output_path', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/7G-S-O',
                        help="Path to save the output results.")
    parser.add_argument('--G_S_video_output_path', type=str,
                        default='/home/ubuntu/End2EndAG-github/AG_person_search/data/8seg_video',
                        help="video Path to save the output results.")

    opt = parser.parse_args()
    print(opt)

    # 1) 空中检测（建议 A_detect_and_save 返回输出视频路径）
    A_detect_and_save(
        A_D_config_file=opt.A_D_config,
        A_D_input_path=opt.A_D_input,
        A_D_output_path=opt.A_D_output,  # 建议是具体的 mp4 文件
        A_D_checkpoint_file=opt.A_D_checkpoint
    )

    # 2) 基于空中检测结果构建 query：用 opt.query_input / opt.query_output
    #    若你希望用空中检测的输出作为输入，则把 opt.query_input 设置为 opt.A_D_output 即可。
    cap = cv2.VideoCapture(opt.query_input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 query_input：{opt.query_input}")

    state = {
        'frame': None,
        'p1': None,
        'save_dir': opt.query_output,
        'count': 0,
        'prefix': '0001_c1s1_50_0'
    }
    os.makedirs(state['save_dir'], exist_ok=True)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', query_get, state)

    while True:
        success, frame = cap.read()
        if not success:
            break
        state['frame'] = frame
        cv2.imshow('image', frame)

        # 空格：下一帧；Enter：结束；q：退出
        key = cv2.waitKey(0)
        if key == 13:  # Enter
            print("标注结束")
            break
        elif key == 32:  # Space
            continue
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 3) 地面 person search
    with torch.no_grad():
        G_person_search(
            opt.cfg,
            opt.data,
            opt.weights,
            images=opt.images,
            Reid_output=opt.ps_output,
            GSI_output=opt.gs_output,
            fourcc=opt.fourcc,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            dist_thres=opt.dist_thres,
            find_class=opt.find_class
        )

    # 4) person-search 后分割
    G_Seg_and_save(opt.G_S_config_file, opt.G_S_checkpoint_file, opt.G_S_input_path, opt.G_S_output_path)

    images_to_video(opt.G_S_output_path, opt.G_S_video_output_path)
