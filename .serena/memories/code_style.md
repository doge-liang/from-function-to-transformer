# 代码风格与规范

## Python 脚本

### 文件结构
```python
#!/usr/bin/env python3
"""模块文档字符串"""

import argparse
import subprocess
from pathlib import Path


def function_name() -> ReturnType:
    """函数文档字符串"""
    pass
```

### 命名规范
| 类型 | 风格 | 示例 |
|------|------|------|
| 常量 | UPPER_CASE | DEFAULT_TIMEOUT |
| 函数 | snake_case | get_project_root() |
| 变量 | snake_case | assets_dir |
| 类 | PascalCase | DiagramGenerator |

### 类型提示
- 必须使用类型提示
- 使用 `list[str]` 而非 `List[str]` (Python 3.10+)
- 返回 `bool` 或 `tuple[bool, str]`

### 导入顺序
1. 标准库 (按字母排序)
2. 无第三方库 (脚本仅用 stdlib)

### 错误处理
```python
try:
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        return False, result.stderr.decode()
    return True, ""
except subprocess.TimeoutExpired:
    return False, "Timeout"
except FileNotFoundError as e:
    return False, f"Tool not found: {e}"
```

### 日志输出
使用 `print()` 配合前缀:
- `[OK]` - 成功
- `[FAIL]` - 失败
- `[ERROR]` - 错误
- `[TIMEOUT]` - 超时

## Markdown 文档

### 数学公式
- 行内: `$E = mc^2$`
- 独立: `$$E = mc^2$$`

### 图表引用
`![](docs/assets/{filename}.svg)`

### 章节结构
- 编号: 01-, 02-, ...
- 长度: 300-500 行

## TikZ 图表

### 文档类
`\documentclass[tikz,border=8pt]{standalone}`

### 常用库
`calc, positioning, fit, backgrounds`

### 样式定义
在 `\tikzpicture` 开始处定义语义化样式名
