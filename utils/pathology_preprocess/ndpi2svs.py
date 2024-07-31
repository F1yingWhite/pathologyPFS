import openslide
import pyvips


def ndpi_to_svs(ndpi_path, svs_path):
    # 打开ndpi文件
    slide = openslide.OpenSlide(ndpi_path)

    # 获取缩略图层次信息
    level_count = slide.level_count

    # 提取全图信息
    width, height = slide.dimensions

    # 创建一个Vips图像对象
    image = pyvips.Image.new_from_file(ndpi_path, access='sequential')

    # 保存为SVS格式
    image.write_to_file(svs_path, Q=99)

    print(f"转换完成: {ndpi_path} -> {svs_path}")


if __name__ == "__main__":
    ndpi_path = 'input.ndpi'  # 输入ndpi文件路径
    svs_path = 'output.svs'  # 输出svs文件路径
    ndpi_to_svs(ndpi_path, svs_path)
