## 快速开始

requirement:

```bash
pip install numpy opencv-python matplotlib
```



命令格式: `python main.py [序列名称] [跟踪算法]`

例如：
```bash
# 交互式运行（程序会提示输入序列名称和跟踪算法）
python main.py

# 使用 MOSSE 算法评估所有序列（批量测试）
python main.py 0 MOSSE

# 使用 KCF 算法评估所有序列
python main.py 0 KCF

# 使用 CSK 算法评估所有序列
python main.py 0 CSK

# 使用 MOSSE 算法评估指定序列（可视化跟踪）
python main.py Basketball MOSSE

# 使用 KCF 算法评估 Bird1 序列
python main.py Bird1 KCF

# 省略算法参数时，默认使用 MOSSE
python main.py Basketball
```


## 数据集

采用 [OTB2015 数据集](https://huggingface.co/datasets/xche32/OTB2015/tree/main) 作为测试基准。

- 位置: 项目根目录下的 `OTB2015/` 文件夹
- 格式要求:
  - 每个序列包含 `img/` 子目录（存放图像帧）
  - 包含 `groundtruth_rect.txt` 文件（真值边界框标注）
  - 标注格式：每行为 `x,y,width,height`（逗号或空格分隔）


## 使用说明

### 批量评估

运行 `python main.py 0 [算法名称]` 进行批量评估：
- 自动遍历 OTB2015 所有序列
- 使用多进程并行加速评估
- 实时输出每个序列的 Precision 和 IoU 指标
- 生成 IoU 曲线图保存到 `output/` 目录
- 支持三种算法：MOSSE、KCF、CSK

### 可视化

运行单个序列时默认启用可视化，批量评估模式（`python main.py 0 [算法]`）自动运行在无显示模式。可视化窗口中，红色框为真值边界框，绿色框为跟踪器预测边界框。

可视化开关位于 `evaluate_sequence` 函数中，设置 `show=True` 或 `False` 可启用或关闭可视化

### 自动退出
`evaluate_sequence` 函数提供两个参数控制自动退出：
- `out_frame_tolerance=114`: 当预测框连续 114 帧离开画面时自动退出，-1 表示不启用该功能
- `zero_iou_tolerance=114`： 当 IoU 值连续 114 帧为 0 时自动退出，-1 表示不启用该功能

自动退出将导致评估提前结束，影响最终指标计算。只是为了节省时间。

## 输出结果

### 单序列模式输出
- 实时可视化窗口 - 显示跟踪过程
- IoU 曲线图 - 保存到 `output/iou_plot_[序列名]_[算法].png`
- 控制台输出 - Precision@20px 和平均 IoU

### 批量评估模式输出
- 控制台输出 - 每个序列的 Precision@20px 和平均 IoU
- IoU 曲线图 - 每个序列的曲线图保存到 `output/iou_[序列名].png`

