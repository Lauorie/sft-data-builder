## Quick Start (Alpha)

```bash
# 1. Clone the repo
git clone https://github.com/huggingface/yourbench.git
cd yourbench

# Use pip install the dependencies
pip install -r requirements.txt

# 3.下载 mineru 需要的模型
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py


python脚本会自动下载模型文件并配置好配置文件中的模型目录

配置文件可以在用户目录中找到，文件名为`magic-pdf.json`

# 4. Run the pipeline with an example config 最好提前配置好 example.yaml
yourbench run --config yourbench/configs/example.yaml --debug
```
