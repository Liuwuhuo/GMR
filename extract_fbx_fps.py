# extract_fbx_fps.py
import bpy
import sys
import os

# 安全提取 -- 之后的参数（兼容路径含空格）
args = sys.argv
if "--" in args:
    fbx_path = args[args.index("--") + 1]
else:
    fbx_path = args[-1]  # 兼容旧版调用

if not os.path.isfile(fbx_path):
    print(f"ERROR: 文件不存在: {fbx_path}")
    sys.exit(1)

# 清空场景并导入
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.import_scene.fbx(filepath=fbx_path, ignore_leaf_bones=True)
except Exception as e:
    print(f"ERROR: FBX 导入失败: {e}")
    sys.exit(1)

# 提取帧率（自动处理 29.97 等 NTSC 分数帧率）
fps = bpy.context.scene.render.fps
fps_base = bpy.context.scene.render.fps_base
real_fps = fps / fps_base
print(f"{real_fps:.4f}")