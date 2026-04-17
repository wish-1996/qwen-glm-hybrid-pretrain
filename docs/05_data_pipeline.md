# 06 数据管线：格式、混合采样、packing 与 mask

## 数据格式

### 1. CSV 图文数据

- **格式**：CSV 文件，包含图像 URL 和文本描述
- **列名**：支持多种列名，如 `url`/`URL` 用于图像路径，`cap_seg`/`text`/`caption` 用于文本描述
- **处理**：
  - 自动处理 BOM 编码
  - 容错处理：缺列、空值等
  - 只使用本地已缓存的图像

### 2. ultrafineweb_zh 文本数据

- **格式**：JSONL 文件，每行一个 JSON 对象，包含 `text` 字段
- **处理**：
  - 容错处理：JSON 解析错误
  - 过滤空文本

## 数据加载

### 1. MultimodalDataset

- **功能**：混合图文数据和文本数据的数据集
- **组成**：
  - `mm_pairs`：图文对列表，每个元素为 (image_path, text)
  - `text_only`：纯文本列表
- **采样策略**：
  - 按比例混合文本数据（默认为 20%）
  - 文本样本随机复用一张真实图片作为占位

### 2. DataLoader 构建

- **参数**：
  - `batch_size`：批次大小
  - `shuffle`：是否打乱数据
  - `num_workers`：数据加载线程数
  - `pin_memory`：是否固定内存
  - `drop_last`：是否丢弃最后不足批次的数据
  - `distributed`：是否使用分布式采样
- **优化**：
  - 工作线程种子设置
  - 内存固定以加速数据传输

## 数据混合采样

### 1. 采样策略

- **图文数据**：从 CSV 文件加载的图文对
- **文本数据**：从 ultrafineweb_zh 加载的纯文本
- **混合比例**：通过 `ultrafineweb_mix_ratio` 参数控制，默认为 0.2（20% 文本数据）

### 2. 实现方式

```python
def _sample_text(self) -> Optional[str]:
    if not self.text_only:
        return None
    if self.cfg.ultrafineweb_mix_ratio <= 0:
        return None
    if random.random() < self.cfg.ultrafineweb_mix_ratio:
        return random.choice(self.text_only)
    return None
```

## 序列打包（Packing）

### 1. 概念

- **定义**：将多个短序列打包成一个长序列，提高训练效率
- **优势**：
  - 减少填充比例
  - 提高 GPU 利用率
  - 加速训练

### 2. 实现思路

- **长度排序**：按序列长度排序，将相似长度的序列打包在一起
- **填充处理**：计算批次中最长序列长度，对其他序列进行填充
- **边界掩码**：创建跨样本边界的掩码，确保模型只关注当前样本

### 3. 示例

```python
def pack_sequences(sequences, max_length):
    # 按长度排序
    sequences.sort(key=lambda x: len(x), reverse=True)
    
    packed = []
    current_pack = []
    current_length = 0
    
    for seq in sequences:
        if current_length + len(seq) <= max_length:
            current_pack.append(seq)
            current_length += len(seq)
        else:
            packed.append(current_pack)
            current_pack = [seq]
            current_length = len(seq)
    
    if current_pack:
        packed.append(current_pack)
    
    return packed
```

## 掩码（Mask）处理

### 1. 注意力掩码

- **作用**：指示模型哪些位置是有效的输入，哪些是填充
- **格式**：二进制掩码，1 表示有效，0 表示填充
- **计算**：
  - 文本部分：根据输入长度计算
  - 图像部分：全 1，因为所有图像补丁都是有效的
  - 拼接：将图像和文本的注意力掩码拼接

### 2. 标签掩码

- **作用**：指示模型哪些位置需要计算损失，哪些需要忽略
- **格式**：-100 表示忽略，其他值表示真实标签
- **计算**：
  - 文本部分：使用输入的 token IDs，填充位置为 -100
  - 图像部分：全 -100，因为图像部分不需要计算损失
  - 拼接：将图像和文本的标签拼接

### 3. 跨样本边界掩码

- **作用**：在序列打包时，确保模型只关注当前样本，不关注其他样本
- **实现**：为每个样本创建独立的掩码，跨样本边界的位置为 0

## 容错处理

### 1. 坏图像处理

- **策略**：跳过坏图像，随机选择其他样本
- **实现**：
  ```python
  pixel_values = _safe_open_image(image_path, self.cfg.image_size)
  if pixel_values is None:
      bad += 1
      if bad > self.cfg.max_bad_samples:
          raise RuntimeError("Too many bad image samples, please check dataset integrity.")
      idx = random.randrange(len(self.mm_pairs))
      continue
  ```

### 2. 数据格式错误处理

- **CSV 错误**：跳过解析错误的行
- **JSON 错误**：跳过解析错误的 JSON 行
- **空值处理**：为缺失的文本提供默认值

### 3. 回退机制

- **作用**：当没有有效样本时，生成合成数据以保证训练能够进行
- **实现**：
  - 创建默认图像（如果没有图像）
  - 使用预设的合成文本

## 性能优化

1. **内存优化**：
   - 使用 `pin_memory` 加速数据传输
   - 批量加载数据

2. **IO 优化**：
   - 使用多线程加载数据
   - 预加载和缓存数据

3. **计算优化**：
   - 图像预处理优化
   - 批量 tokenization

## 下一步优化

1. **流式读取**：使用 webdataset/parquet/arrow 流式读取大型数据集
2. **更高效的图像解码**：使用 opencv/accimage/turbojpeg
3. **动态批处理**：根据序列长度动态调整批次大小
4. **数据增强**：添加图像和文本的数据增强策略
