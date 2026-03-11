import numpy as np
import argparse
import os

# 目标文件路径
TARGET_YAML_PATH = "/cluster/project/cvg/students/shangwu/FrontierNet/config/hm3d_exploration_1.yaml"

def generate_matrix(x, y, z, angle_deg):
    """
    生成水平视线的 World-to-Camera 矩阵 (4x4)
    """
    theta = np.deg2rad(angle_deg)
    
    # 1. 构建相机坐标轴 (在世界坐标系下)
    # Z轴 (Forward): 指向视线方向 (XY平面)
    cam_z_w = np.array([np.cos(theta), np.sin(theta), 0.0])
    
    # Y轴 (Down): 垂直指向地面 -Z
    cam_y_w = np.array([0.0, 0.0, -1.0])
    
    # X轴 (Right): Y x Z
    cam_x_w = np.cross(cam_y_w, cam_z_w)
    
    # 2. 组装旋转矩阵 R_wc (World-to-Camera)
    # R_wc 的每一行是相机的坐标轴
    R_wc = np.array([cam_x_w, cam_y_w, cam_z_w])
    
    # 3. 计算平移向量 t = -R * C
    C = np.array([x, y, z])
    t = -np.dot(R_wc, C)
    
    # 4. 组装 4x4 矩阵
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t
    
    return T_wc

def format_matrix_lines(matrix):
    """
    将矩阵转换为 YAML 格式的字符串列表
    """
    lines = []
    for row in matrix:
        # 格式: "  - [0.000, 0.000, ...]"
        # 保持与原文件一致的缩进 (2个空格)
        line_str = f"  - [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}]\n"
        lines.append(line_str)
    return lines

def update_yaml_file(file_path, new_matrix_lines):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    with open(file_path, 'r') as f:
        content = f.readlines()

    # --- 智能查找逻辑 ---
    key_line_index = -1
    start_replace_index = -1

    # 1. 找到 "initial_cam_extrinsic:"
    for i, line in enumerate(content):
        if "initial_cam_extrinsic:" in line:
            key_line_index = i
            break
    
    if key_line_index == -1:
        print("错误: 在文件中未找到 'initial_cam_extrinsic:' 关键字")
        return

    # 2. 寻找其后的数据行（跳过注释）
    # 从关键字下一行开始找，找到第一个以 '-' 开头的行
    for i in range(key_line_index + 1, len(content)):
        stripped = content[i].strip()
        if stripped.startswith("- ["):
            start_replace_index = i
            break
        # 如果遇到其他 key (不含 '-') 且不为空，说明找过头了
        if stripped and not stripped.startswith("#") and not stripped.startswith("-"):
             print("错误: 未找到矩阵数据行，可能格式不匹配")
             return

    if start_replace_index != -1:
        print(f"定位成功: 将替换第 {start_replace_index+1} 到 {start_replace_index+4} 行")
        
        # 3. 执行替换 (替换4行)
        # 检查我们要替换的行数是否足够
        if start_replace_index + 4 <= len(content):
            for offset in range(4):
                content[start_replace_index + offset] = new_matrix_lines[offset]
            
            # 4. 写回文件
            with open(file_path, 'w') as f:
                f.writelines(content)
            print("文件更新成功！")
        else:
            print("错误: 文件剩余行数不足以替换矩阵")
    else:
        print("错误: 未能定位到矩阵数据的起始行")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成相机矩阵并更新 YAML 配置文件")
    parser.add_argument("x", type=float, help="世界坐标 X")
    parser.add_argument("y", type=float, help="世界坐标 Y")
    parser.add_argument("z", type=float, help="世界坐标 Z")
    parser.add_argument("angle", type=float, help="水平视角角度 (0=X轴, 90=Y轴)")
    
    args = parser.parse_args()

    # 1. 计算矩阵
    matrix = generate_matrix(args.x, args.y, args.z, args.angle)
    
    # 2. 格式化数据
    new_lines = format_matrix_lines(matrix)
    
    # 3. 打印到控制台 (预览)
    print("-" * 30)
    print(f"生成的矩阵 (Pos: {args.x},{args.y},{args.z} | Ang: {args.angle}):")
    for line in new_lines:
        print(line.strip())
    print("-" * 30)

    # 4. 更新文件
    print(f"正在更新文件: {TARGET_YAML_PATH}, with: {new_lines}")
    update_yaml_file(TARGET_YAML_PATH, new_lines)