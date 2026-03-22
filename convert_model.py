"""
模型转换脚本：PyTorch (.pt) -> ONNX -> NCNN
用于将 YOLOv8-Pose 模型转换为 Android 端可用的 NCNN 格式
"""
import os
import subprocess
import sys

def export_to_onnx(model_path: str, output_dir: str = "./android/app/src/main/assets"):
    """
    将 PyTorch 模型导出为 ONNX 格式
    """
    print("=" * 60)
    print("Step 1: Exporting PyTorch model to ONNX...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        # 导出 ONNX
        onnx_path = model.export(
            format='onnx',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False
        )
        
        print(f"ONNX model exported to: {onnx_path}")
        
        # 移动到目标目录
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, "yolov8n-pose.onnx")
        shutil.copy(onnx_path, target_path)
        print(f"ONNX model copied to: {target_path}")
        
        return target_path
        
    except Exception as e:
        print(f"Error exporting ONNX: {e}")
        raise


def onnx_to_ncnn(onnx_path: str, output_dir: str = "./android/app/src/main/assets"):
    """
    将 ONNX 模型转换为 NCNN 格式
    需要安装 onnx2ncnn 工具
    """
    print("=" * 60)
    print("Step 2: Converting ONNX to NCNN...")
    print("=" * 60)
    
    param_path = os.path.join(output_dir, "yolov8n-pose.param")
    bin_path = os.path.join(output_dir, "yolov8n-pose.bin")
    
    # 方法 1: 使用 onnx2ncnn 命令行工具
    try:
        result = subprocess.run(
            ["onnx2ncnn", onnx_path, param_path, bin_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"NCNN param saved to: {param_path}")
            print(f"NCNN bin saved to: {bin_path}")
            return param_path, bin_path
        else:
            print(f"onnx2ncnn failed: {result.stderr}")
            
    except FileNotFoundError:
        print("onnx2ncnn not found in PATH")
    
    # 方法 2: 使用 Python onnx-simplifier
    print("\nTrying alternative method with onnx-simplifier...")
    try:
        import onnx
        from onnxsim import simplify
        
        # 加载并简化 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        simplified, _ = simplify(onnx_model)
        
        simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
        onnx.save(simplified, simplified_path)
        print(f"Simplified ONNX saved to: {simplified_path}")
        
        print("\nPlease manually convert to NCNN using one of these methods:")
        print("1. Download onnx2ncnn from: https://github.com/Tencent/ncnn/releases")
        print(f"   Command: onnx2ncnn {simplified_path} {param_path} {bin_path}")
        print("2. Use online converter: https://convertmodel.com/")
        print("3. Build ncnn tools from source:")
        print("   git clone https://github.com/Tencent/ncnn.git")
        print("   cd ncnn && mkdir build && cd build")
        print("   cmake .. && make")
        print(f"   ./tools/onnx/onnx2ncnn {simplified_path} {param_path} {bin_path}")
        
        return None, None
        
    except ImportError:
        print("onnx-simplifier not installed. Install with: pip install onnx-simplifier")
        return None, None


def optimize_ncnn_for_mobile(param_path: str, bin_path: str):
    """
    优化 NCNN 模型以适应移动端
    需要使用 ncnnoptimize 工具
    """
    print("=" * 60)
    print("Step 3: Optimizing NCNN for mobile...")
    print("=" * 60)
    
    opt_param = param_path.replace(".param", "_opt.param")
    opt_bin = bin_path.replace(".bin", "_opt.bin")
    
    try:
        result = subprocess.run(
            ["ncnnoptimize", param_path, bin_path, opt_param, opt_bin, "0"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"Optimized NCNN param: {opt_param}")
            print(f"Optimized NCNN bin: {opt_bin}")
            
            # 替换原文件
            import shutil
            shutil.move(opt_param, param_path)
            shutil.move(opt_bin, bin_path)
            print("Optimized files replaced original files")
        else:
            print(f"ncnnoptimize failed: {result.stderr}")
            
    except FileNotFoundError:
        print("ncnnoptimize not found, skipping optimization")
        print("You can optimize later with: ncnnoptimize input.param input.bin output.param output.bin 0")


def main():
    print("YOLOv8-Pose Model Conversion Script")
    print("=" * 60)
    
    # 模型路径
    model_path = "./yolov8n-pose.pt"
    output_dir = "./android/app/src/main/assets"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please ensure yolov8n-pose.pt is in the current directory")
        sys.exit(1)
    
    # Step 1: PT -> ONNX
    onnx_path = export_to_onnx(model_path, output_dir)
    
    # Step 2: ONNX -> NCNN
    param_path, bin_path = onnx_to_ncnn(onnx_path, output_dir)
    
    # Step 3: 优化 (可选)
    if param_path and bin_path:
        optimize_ncnn_for_mobile(param_path, bin_path)
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")
    print("- yolov8n-pose.param  (NCNN model structure)")
    print("- yolov8n-pose.bin    (NCNN model weights)")
    print("\nNext steps:")
    print("1. Copy model files to android/app/src/main/assets/")
    print("2. Download NCNN SDK and place in android/app/src/main/jni/ncnn/")
    print("3. Build and run the Android app")


if __name__ == "__main__":
    main()
