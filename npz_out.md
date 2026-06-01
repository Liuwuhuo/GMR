# npz -> extend_npz

# 工程
```bash 
git clone https://github.com/robfiras/loco-mujoco.git
```

## 按照工程下的readme配置环境和基本安装、

## 脚本放在工程目录下


## npz后处理
```bash
# 基本用法
python convert_custom_format_and_extend_new.py \
    -i motion_data/input/xxx.npz \
    -o motion_data/output/xxx_50hz_extended.npz \
    -of 50

python convert_custom_format_and_extend_new.py \
  --batch \
  -i path/to/input_dir \
  -o path/to/output_dir \
  -of 50
```