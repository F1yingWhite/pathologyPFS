import os
import glob

if __name__ == '__main__':
    path = "../../data/SVS/转移瘤2SVS"
    new_path = "../../data/tiles/Path"
    for file in glob.glob(os.path.join(path, "*.svs")):
        file_new_path = os.path.join(new_path, os.path.basename(file))
        # 使用命令行调用
        os.system(
            f"python sample_tiles.py --input_slide {file} --output_dir {file_new_path} --tile_size 128 --n 204800 --out_size 224"
        )
