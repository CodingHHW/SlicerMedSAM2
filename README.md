# SlicerMedSAM2

SlicerMedSAM2 是一个基于 3D Slicer 平台开发的医学图像分割插件，集成了 MedSAM2 模型，提供高效、准确的 2D 交互式分割功能。

## 功能特点

### 核心功能
- **2D 交互式分割**：支持通过点和边界框进行交互式分割
- **正负样本提示**：可灵活切换正样本和负样本提示模式
- **服务器连接**：支持连接到本地或远程服务器进行模型推理
- **分割编辑**：集成了 Slicer 的分割编辑器，方便进行精细调整

### 交互工具
- **点工具**：通过点击添加分割提示点
- **边界框工具**：通过拖拽绘制边界框进行分割初始化

## 安装说明

### 前提条件
- 已安装 3D Slicer 4.11 或更高版本
- Python 3.6 或更高版本

### 安装方法

#### 方法 1：通过扩展管理器安装（推荐）
1. 打开 3D Slicer
2. 点击菜单栏的 "Extensions" -> "Extension Manager"
3. 在搜索框中输入 "SlicerMedSAM2"
4. 点击 "Install" 按钮进行安装
5. 重启 Slicer 以启用插件

#### 方法 2：手动安装
1. 克隆或下载本仓库：
   ```bash
   git clone https://github.com/CodingHHW/SlicerMedSAM2.git
   ```
2. 将 SlicerMedSAM2 目录复制到 Slicer 的扩展目录中
3. 重启 Slicer 以启用插件

## 使用方法

### 基本操作流程

1. **加载图像**：在 Slicer 中加载需要分割的医学图像
2. **打开插件**：在模块浏览器中找到 "MedSAM2" 模块并点击打开
3. **选择交互工具**：
   - 点击 "Point" 按钮进入点交互模式
   - 或点击 "Bounding Box" 按钮进入边界框交互模式
4. **设置提示类型**：
   - 点击 "Positive" 设置为正样本提示
   - 或点击 "Negative" 设置为负样本提示
5. **添加分割提示**：
   - 在图像上点击添加点提示
   - 或拖拽绘制边界框
6. **查看分割结果**：分割结果将实时显示在图像上
7. **编辑分割**：使用 "Segment Editor" 面板进行精细调整
8. **重置或继续**：
   - 点击 "Reset segment" 重置当前分割
   - 或点击 "Next segment" 继续分割下一个结构

### 服务器配置

1. 在 "Configuration" 标签页中，设置服务器 URL
2. 点击 "Test Connection" 按钮测试服务器连接
3. 确保服务器正在运行且可访问

## 项目结构

```
SlicerMedSAM2/
├── CMakeLists.txt          # CMake 配置文件
├── MedSAM2/                # 插件主目录
│   ├── CMakeLists.txt      # MedSAM2 模块 CMake 配置
│   ├── MedSAM2.py          # 插件主代码文件
│   ├── Resources/          # 资源目录
│   │   ├── Icons/          # 图标文件
│   │   │   └── MedSAM2.png # 插件图标
│   │   └── UI/             # UI 界面文件
│   │       └── MedSAM2.ui  # Qt Designer 设计的 UI 文件
│   └── Testing/            # 测试目录
│       └── Python/         # Python 测试文件
├── SlicerMedSAM2.png       # 插件缩略图
├── .gitignore              # Git 忽略文件配置
└── README.md               # 项目说明文档
```

## 开发说明

### 环境搭建
1. 安装 3D Slicer
2. 安装 Python 开发环境
3. 克隆本仓库

### 编译和测试
1. 使用 CMake 配置项目
2. 编译项目
3. 运行测试：
   ```bash
   python -m pytest MedSAM2/Testing/Python -v
   ```

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目！

### 提交代码
1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 致谢

- 感谢 3D Slicer 团队提供的强大平台
- 感谢 MedSAM2 模型的开发者
- 感谢所有为项目做出贡献的开发者

## 联系方式

- 项目地址：[https://github.com/CodingHHW/SlicerMedSAM2](https://github.com/CodingHHW/SlicerMedSAM2)
- 作者：CodingHHW

## 更新日志

### v0.1.0 (2026-01-19)
- 初始版本
- 实现了基本的 2D 交互式分割功能
- 支持点和边界框交互
- 集成了分割编辑器
- 支持服务器连接配置

## 已知问题

- 目前仅支持 2D 分割
- 部分特殊图像格式可能存在兼容性问题
- 大图像分割速度有待优化

## 未来计划

- 支持 3D 分割
- 优化分割速度
- 添加更多交互工具
- 支持离线推理
- 增加模型选择功能
- 改进用户界面

---

**SlicerMedSAM2 - 让医学图像分割更简单、更高效！**