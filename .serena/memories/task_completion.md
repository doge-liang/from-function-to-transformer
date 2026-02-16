# 任务完成检查清单

## 修改代码后
1. 运行相关脚本验证功能
2. 检查控制台输出是否正常

## 修改 TikZ 图表后
1. 运行 `python scripts/generate-tikz.py <file.tex>`
2. 验证 `docs/assets/` 中的 SVG 文件
3. 在 Markdown 中检查图表显示

## 修改 Markdown 文档后
1. 运行 `npm run lint:md`
2. 如有问题，运行 `npm run lint:md:fix`
3. 在 Markdown 预览中检查渲染效果

## 提交前
1. 重新生成所有修改过的图表
2. 运行 `npm run lint:md`
3. 验证 SVG 显示正确
4. 确保 `docs/assets/` 包含所有新 SVG
5. 确保没有临时 LaTeX 文件 (*.aux, *.log, *.dvi)

## 无需运行
- 无正式测试套件
- 无 CI/CD 流程
- Python 无 linter (手动遵循风格)
