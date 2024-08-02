import os
import glob
import concurrent.futures


def process_file(file):
    file_new_path = os.path.join(new_path, os.path.basename(file))
    os.system(
        f"python sample_tiles.py --input_slide {file} --output_dir {file_new_path} --tile_size 128 --n 204800 --out_size 224"
    )


if __name__ == '__main__':
    path = "../../data/LBM_Path/转移瘤ES-SVS"
    new_path = "../../data/tiles/转移瘤ES-SVS"
    files = glob.glob(os.path.join(path, "*.svs"))

    # 使用线程池执行并行任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_file, files)
