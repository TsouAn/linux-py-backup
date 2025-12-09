#!/usr/bin/env python3
import sys
import os
import glob
import cv2
import numpy as np
from edge_impulse_linux.runner import ImpulseRunner

def main():
    if len(sys.argv) != 3:
        print("使用方式: python3 classify_od.py <model.eim> <圖片>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print(f"載入模型: {model_path}")
    print(f"載入圖片: {image_path}")

    # 初始化推論引擎
    runner = ImpulseRunner(model_path)
    try:
        model_info = runner.init()
        print(f"模型標籤: {model_info['model_parameters']['labels']}")

        # 取得模型需要的輸入尺寸
        width = model_info['model_parameters']['image_input_width']
        height = model_info['model_parameters']['image_input_height']
        

        # 讀取並前處理圖片
        img = cv2.imread(image_path)


        # 1. 轉成 RGB (因為 OpenCV 預設是 BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 轉成 灰階 (這是模型需要的！)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 3. 縮放
        img_resized = cv2.resize(img_gray, (width, height))
 
        # 4. 轉換資料格式
        # 注意：有些模型需要 0-1 的數值，有些需要 0-255。
        # 這裡保持原始數值 (0-255)，這是 Edge Impulse Linux Runner 的標準做法
        img_float = img_resized.astype('float32')
        img_processed = img_float.flatten()

        # 執行推論
        result = runner.classify(img_processed)

        # --- 除錯步驟：印出原始結果 ---
        print("\n=== 原始回傳資料 ===")
        print(result)
        print("==================\n")
        # ------------------------

        # 顯示結果
        # 顯示結果
        if 'bounding_boxes' in result['result']:
            for i, box in enumerate(result['result']['bounding_boxes']):
                print(f"物件 {i+1}: {box['label']} "
                    f"({box['value']:.2f})")
    finally:
        runner.stop()

if __name__ == "__main__":
    main()
