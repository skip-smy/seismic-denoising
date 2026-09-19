import segyio
import numpy as np
import os

def split_segy_by_trace_ratio(input_file, output_folder,output1_name, output2_name, ratio=(0.5, 0.5)):
    """
    按道数比例分割SEGY文件
    :param input_file: 输入SEGY文件路径
    :param output1_file: 第一个输出文件路径（比例1）
    :param output2_file: 第二个输出文件路径（比例2）
    :param ratio: 分割比例，如(0.2, 0.8)表示2:8分割
    """
    output1_file = os.path.join(output_folder, output1_name)
    output2_file = os.path.join(output_folder, output2_name)
    with segyio.open(input_file, "r",ignore_geometry=True) as src:
        # 获取总道数
        total_traces = src.tracecount
        print(f"总道数: {total_traces}")

        # 计算分割点（四舍五入到整数道）
        split_point = int(total_traces * ratio[0])
        print(
            f"分割点: 第{split_point}道（前{split_point}道为第一个文件，剩余{total_traces - split_point}道为第二个文件）")

        # 复制SEGY规格和头部信息
        spec = segyio.spec()
        spec.format = src.format
        spec.samples = src.samples
        spec.tracecount = split_point
        # 创建第一个文件（20%道数）
        with segyio.create(output1_file, spec) as dst1:
            dst1.text[0] = src.text[0]  # 复制文本卷头
            dst1.bin = src.bin  # 复制二进制卷头
            for i in range(split_point):
                dst1.trace[i] = src.trace[i]  # 复制道数据
                dst1.header[i] = src.header[i]  # 复制道头信息

        # 创建第二个文件（80%道数）
        spec2 = segyio.spec()
        spec2.format = src.format
        spec2.samples = src.samples
        spec2.tracecount = total_traces - split_point

        with segyio.create(output2_file, spec2) as dst2:
            dst2.text[0] = src.text[0]  # 复制文本卷头
            dst2.bin = src.bin  # 复制二进制卷头
            for i, trace_idx in enumerate(range(split_point, total_traces)):
                dst2.trace[i] = src.trace[trace_idx]  # 复制道数据
                dst2.header[i] = src.header[trace_idx]  # 复制道头信息

    print(f"分割完成！")
    print(f"文件1: {output1_file} ({split_point}道，占比{ratio[0] * 100}%)")
    print(f"文件2: {output2_file} ({total_traces - split_point}道，占比{ratio[1] * 100}%)")


# 使用示例
if __name__ == "__main__":
    input_segy = "./data/Test/test.segy"
    output_folder = "./data"
    output1 = "val.segy"
    output2 = "test-2.segy"
    split_segy_by_trace_ratio(input_segy, output_folder,output1, output2, ratio=(0.5, 0.5))