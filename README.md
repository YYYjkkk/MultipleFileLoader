# MultipleFileLoader
## 项目来历
本仓库的内容仅仅是提取了Langchain-Chatchat项目中FileLoader的内容，并做了一些微小改动，能够进行jpg、docx、pdf、ppt到txt格式的转换方便使用
## 项目结构
`input/`为输入的文件夹
`output/`为输出的txt文件夹
其他为对应的文件加载py脚本
## 注意：
在pdf的转换中实现了能区分两列pdf论文并按照先左后右的顺序进行识别的改动，也可以自动实现单列文本的识别
