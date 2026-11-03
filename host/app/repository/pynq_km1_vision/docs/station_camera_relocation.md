# 自动恢复机位与人工给定高度

本入口把镜头参数、相机当前姿态、固定工位和已有机械臂局部映射分开保存。目标是在相机移位后自动重新计算物体坐标，减少重新摆卡和逐点对准。当前为完整离线工具，不采集相机、不联网、不打开串口，也不修改现场执行器或生产配置。

现场优先使用[一页引导采图与三类资料复用](station_guided_capture.md)：复用现有只读相机桥接，停稳自动拍照、自动筛选历史图并计算，同时拍摄固定标记与旧棋盘即可登记坐标。下文手工照片列表和直尺对应点方式保留为离线/备用入口。

已有标定可以用于计算和拟合训练。此次训练是平面映射与机械臂局部仿射拟合，不是重新训练YOLO，也不是从少量历史记录学出任意物体的完整轨迹。实际复用结果见[2026-10-29记录](../../docs/team/reviews/2026-10-29-station-camera-reuse.md)。

## 三层坐标与高度

- `station-camera-v1`：相机ID、分辨率、固定焦距/对焦档案ID、内参矩阵和畸变。换相机或改变镜头设置使用新的档案；已有单应矩阵不能直接当镜头内参。
- `station-markers-v1`：固定标记各角点在工位中的毫米坐标。+Z朝桌面上方，纸面+X向右、+Y向上；检测角点顺序是编码标记的左上、右上、右下、左下。整块标记板必须与机械臂底座保持固定关系。
- `station-legacy-profile-v1`：已有纸卡到机械臂名义XY的映射、样件高度、模型Z及已核对点域。旧的图像单应矩阵只作为源资料保留，不套用到新机位。标记工位与旧纸卡原点C2之间还需要一次对应关系。

人工输入的是**所选可见物面点离桌面的实际高度**，单位mm，不是机械臂模型Z，也不一定是物品最大高度。软件对像素做畸变校正，再用当前相机姿态计算视线与该高度平面的交点。瓶罐曲面上选择不同点时表面高度也可能不同，不能把整个掩码都当作顶面平面。

几何计算可以处理配置声明范围内的新高度；旧机器人档案只允许其原高度与已核对点域。53毫米塑料瓶不能自动继承56毫米金属罐的机械臂夹持参数，也不能用56/65毫米两点插值推断任意夹取高度、腕角、夹爪力度或轨迹。以后补深度相机可替换高度来源；新相机仍需要自身内参和相对固定工位的定位。

## 复用现有数据与训练

工程目录运行：

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration reuse --manifest ../local-only/calibration-work/reuse-manifest.json --output-dir ../local-only/calibration-work/reused-new
```

manifest中所有路径相对manifest所在目录。结构：

```json
{
  "planes": [
    {"path": "records/card53.json", "surface_height_mm": 53, "square_mm": 20}
  ],
  "profiles": [
    {"path": "records/same-can.json", "profile_id": "same-can-56mm"}
  ],
  "robot_fits": [
    {"name": "can56-grip", "path": "records/can56-pairs.json"}
  ]
}
```

平面输入必须有A1～E4共20角点、C2原点、+X向右/+Y向上的明确约定。20毫米是本队记录中已有的格距，换纸卡须填实际测量值。每张图独立采用预先固定的12个拟合点和8个保留检查点，不合并不同机位/高度。输出原图哈希、源JSON哈希、像素覆盖域和纸卡范围；检查是同一张图内的几何一致性，不能当作独立新样件或机械臂精度。

机器人训练输入包括 `card_xy_mm`、`nominal_xy_mm`、`check_card_xy_mm`、`check_nominal_xy_mm`、`object_height_mm`、`fixture_height_mm`、`model_z_mm`、`alpha_deg`。至少3个非共线拟合点与1个独立位置检查点。输出明确保留 `physical_error_quantitatively_known=false`；人工“正对”和取整坐标拟合的接近零残差不能称零机械误差。物体高度与当时卡片/夹爪所在高度分别记录，不混用。

所有输入按原始字节复制到新的`sources/`目录并记录哈希；原文件不修改。输出目录必须不存在。

## 固定标记与镜头内参

现成可打印文件：

- [A4固定标记板](../data/station_calibration/fixed_markers_a4.svg)：ID10～13，每枚黑色外边长45毫米，带100毫米检尺；不要剪开后随意重新摆放。
- [镜头标定棋盘](../data/station_calibration/intrinsics_chessboard_a4.svg)：5×4内角点、20毫米格距。也可使用现有相同格距的完整棋盘。
- [标记板设计坐标](../config/station_markers.design.json)与[尚缺内参的相机示例](../config/station_camera.pending.example.json)。设计坐标不是已安装/已验收工位，内参示例故意为null，不填猜测值。

按100%实际尺寸打印，关闭适应页面，实测检尺和黑边尺寸后再固定到刚性平板。相机需能看见至少3枚分布充分的标记，避免机械臂遮挡；视角、像素尺寸、反光和打印变形都影响定位，45毫米只是此版图纸尺寸，不是所有机位的通用最优大小。设计文件工作点域与高度范围也需要按实际工位核对，不能据设计声明当作机器人已验收范围。

也可重新生成一套离线资料：

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration kit --output-dir ../local-only/calibration-work/kit-new
```

一次镜头内参计算使用manifest，包含 `camera_id`、`lens_profile_id`、`image_size`、`pattern_inner_corners=[5,4]`、`square_mm=20`、`fit_images`和`check_images`。图像路径相对manifest所在目录。至少12张拟合照片及3张未参与内参拟合的检查照片，保持同一相机/分辨率/焦距/对焦设置，让棋盘覆盖不同位置、距离和倾角。程序检查棋盘中心横纵覆盖至少图像尺寸的20%、估计法向跨度至少15度、拟合和检查重投影误差。原有只改变垫高、基本平行的几张标定卡图不能直接冒充这组数据。误差通过也不等同实际尺寸定位精度验收。

CLI缺省仍为12张拟合和`brown5`镜头模型。高效引导页显式使用`minimum_fit_images=10`、`require_leave_one_out_stability=true`及`calibration_model=brown4`，从10张拟合/3张检查起步，并逐张留一重拟合，要求焦距变化≤2.5%、主点变化≤8像素；其余覆盖、倾角、拟合/检查误差门槛仍保留。brown4保留k1/k2/p1/p2、固定k3=0；全量和所有留一计算均使用同一约束。质量未通过时先检查原因，不再自动要求每次增加3张；检查照片不移作训练。允许的最小拟合张数为10，不能只改数字绕过数据不足。

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration intrinsics --manifest ../local-only/calibration-work/intrinsics-manifest.json --output ../local-only/calibration-work/camera.json
```

当前环境OpenCV含ArUco，无需为本轮新增依赖；新环境需要支持`cv2.aruco.ArucoDetector`的OpenCV。仓库已有OpenCV依赖；不同时安装多个冲突的cv2包。

## 自动定位与高度计算

先对同一停稳机位的至少3张独立录制图定位：

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration localize --images ../local-only/calibration-work/a.png ../local-only/calibration-work/b.png ../local-only/calibration-work/c.png --camera ../local-only/calibration-work/camera.json --station ../local-only/calibration-work/station.json --output ../local-only/calibration-work/pose.json
```

使用多标记平面PnP、备选解歧义检查及重投影优化；要求至少3枚已知标记、足够像素覆盖、角点重投影RMS≤0.8像素/最大≤1.5像素、三帧相机中心差≤2毫米且姿态差≤0.5度。这些是候选软件筛选阈值，不是已经实测的定位精度。拒绝重复图、标记ID冲突、分辨率改变、显著位移、退化/歧义解或标记太小；相机及标记配置哈希绑定到输出。

给定像素与人工高度计算毫米坐标：

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration point --camera ../local-only/calibration-work/camera.json --station ../local-only/calibration-work/station.json --pose ../local-only/calibration-work/pose.json --pixel 320 180 --height-mm 56 --output ../local-only/calibration-work/point.json
```

这里只是命令格式；像素必须来自当前对应图像中所选物面点，不是示例抓点。输出世界坐标，检查高度范围、射线方向和工位点域。库函数`world_to_pixels`可把同一工位内毫米区域投影回当前图像，供后续迁移取物区/箱区使用；本轮未替换现有固定像素ROI。

旧纸卡C2与固定板坐标的连接采用至少3组非共线毫米对应点和2组独立检查点。输入 `world_frame_id`、`world_xy_mm`、`card_xy_mm`、`check_world_xy_mm`、`check_card_xy_mm`，运行：

```powershell
..\.venv\Scripts\python.exe -m src.station_calibration register --pairs ../local-only/calibration-work/world-card-pairs.json --output ../local-only/calibration-work/world-card-registration.json
```

只拟合刚性旋转/平移，拒绝镜像、用缩放掩盖毫米单位错误和拟合/检查位置重复；几何残差须≤1毫米。这个检查只验证所填对应点的数学一致性，原机械臂误差仍沿用旧记录限制。随后`point`可额外传入 `--profile`及`--registration`输出旧档案下的名义抓取/悬停坐标。未登记对应关系、样件高度不符或超出旧点域均拒绝，不自动放大工作范围。

## 实物接入边界

引导助手现已支持首次登记后仅固定标记的实时自动机位恢复，无需旧棋盘或手动按钮。`src.station_relocalization`提供三帧/新鲜度/遮挡/机位变化保护、三类原点域及高度投影、只读`/localization`输出。正式受监督单件入口也已[接入新坐标、区域和基线](sorting_relocated_grasp.md)，软件及只读实际PL核对通过；本轮尚未执行该入口的机械抓放，不等同新的物理精度或无人值守验收。

全部输出带`automatic_motion_allowed=false`；相机姿态结果为历史图回放，不能证明实时相机新鲜性。没有命令生成、串口、模型切换、PL加载或运动执行参数。现有冻结计划执行器也拒绝读取这些不同kind的文件。

后续接入实时闭环仍须绑定真实帧、机位版本和毫米工位区域；相机在动作中移动仍应停止，停稳后重新定位，再核对场景和图像证据。不能放宽旧机位检查、几何配准旧照片或沿用旧像素ROI来隐藏相机变化。固定标记板或机械臂底座发生相对移动时，旧工位登记不能复用。

参考：[OpenCV标记板姿态估计](https://github.com/opencv/opencv/blob/5.x/doc/tutorials/objdetect/aruco_board_detection/aruco_board_detection.markdown)、[相机标定与PnP](https://docs.opencv.org/4.x/dc/d43/tutorial_camera_calibration_square_chess.html)。
