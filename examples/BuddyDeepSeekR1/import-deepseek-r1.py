#!/usr/bin/env python3
# ===- import-deepseek-r1.py ---------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# 这是 DeepSeekR1 模型的导入测试脚本。
#
# ===---------------------------------------------------------------------------

import argparse
import os

import numpy
import torch

# Buddy 前端 API，用于把 PyTorch 模型捕获成 Buddy 图并生成 MLIR。
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_matmul_transpose_reshape,
    eliminate_transpose,
    flash_attention_prefill,
    gqa_attention_fusion,
    simply_fuse,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa

# TorchInductor 的分解规则用于把 Dynamo 捕获到的算子拆成更容易降低的形式。
from torch._inductor.decomposition import decompositions as inductor_decomp

# HuggingFace 模型加载器，以及静态 KV cache 辅助类。
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)

# 命令行参数用于控制输出目录和导出精度。
parser = argparse.ArgumentParser(description="DeepSeekR1 Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f32",
    choices=["f32", "f16", "bf16"],
    help="Precision mode for MLIR/input data. Choose from %(choices)s.",
)
args = parser.parse_args()

# 提前创建输出目录，避免后续写文件时目录不存在。
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 优先使用本地模型路径；如果未设置环境变量，则使用公开的 HuggingFace 模型名。
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 按请求的精度加载模型，并切换到推理模式。
if args.precision == "f16":
    model = (
        # from_pretrained(model_path, dtype=torch.float16)：从本地路径或 HuggingFace 加载预训练因果语言模型，
        # 并按 float16 类型加载权重。最终得到的是一个用于推理和后续图捕获的 FP16 模型。
        AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
        .eval() # 切换到推理模式，关闭 Dropout 等训练行为。
        .half() # 将模型参数和部分 buffer 转换为 float16。
    )
elif args.precision == "bf16":
    model = (
        # 加载模型，并使用 BF16 类型的权重。BF16 与 FP16 都是 16 位浮点数，
        # 但 BF16 的指数位更多，表示范围更大，通常更适合大模型推理。
        AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        .eval()
        .bfloat16()
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).eval()
# 这里先关闭默认 cache，因为脚本会显式导出 prefill 和 decode 两条路径。
model.config.use_cache = False

# 创建两个 importer：一个用于 prefill，一个用于 decode。
# 创建一个配置了 TOSA 映射和算子分解规则的 PyTorch 图导入器；
# 后续将使用 Prefill 示例输入调用它，并把生成函数命名为 forward_prefill。
prefill_func_name = "forward_prefill" # 指定生成的 MLIR 入口函数名称为 forward_prefill。
dynamo_compiler_prefill = DynamoCompiler(
    # 优先使用 TOSA 算子映射，将 Buddy Graph 算子转换为 TOSA 等 MLIR 操作。
    primary_registry=tosa.ops_registry,
    # 使用 TorchInductor 的分解规则，将复杂 PyTorch 算子拆成更基础、易于转换的算子。
    aot_autograd_decomposition=inductor_decomp,
    # 将生成函数命名为 forward_prefill。
    func_name=prefill_func_name,
)
# 创建一个配置了 TOSA 映射和算子分解规则的 PyTorch 图导入器；
# 后续将使用 Decode 示例输入调用它，并把生成函数命名为 forward_decode
dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

"""
Python 的 with 语句用于管理需要“进入”和“退出”处理的资源或运行环境，例如文件、锁、数据库连接和 PyTorch 梯度状态。

with open("data.txt", "r") as file:
    content = file.read()
执行过程相当于：
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()
"""
# 使用构造的示例输入追踪模型，并导入捕获到的计算图。
with torch.no_grad():
    # 处理 f16 模式下的模型图捕获，主要分为 Prefill 和 Decode 两条路径。
    if args.precision == "f16":
        # StaticCache 固定 KV cache 布局，让 importer 看到稳定的张量形状。
        past_key_values_prefill = StaticCache(
            config=model.config, max_cache_len=1024
        )
        past_key_values_decode = StaticCache(
            config=model.config, max_cache_len=1024
        )
        # 构造用于图捕获的示例输入：
        data_prefill = { # Prefill：batch size 为 1，序列长度为 1024。
            "input_ids": torch.zeros((1, 1024), dtype=torch.int64),
        }
        data_decode = { # Decode：batch size 为 1，每次只输入 1 个 token。
            "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        }

        # 表示 Decode 时当前 token 要写入 KV Cache 的位置。这里的 200 是捕获时使用的示例位置，真正运行时通常由外部程序传入。
        cache_position = torch.tensor([200], dtype=torch.int64)

        # 调用前面创建的 DynamoCompiler，捕获长度为 1024 的 Prefill 路径。
        graphs_prefill = dynamo_compiler_prefill.importer(
            model,
            input_ids=data_prefill["input_ids"],
            use_cache=True, # 要求模型生成并使用 KV Cache。
            # past_key_values=past_key_values_prefill, 参数当前被注释掉，因此 Prefill 不显式传入之前的 Cache。
            cache_implementation="static", # 使用静态布局的 Cache，方便编译器获得稳定形状。
        )
        # 先实际执行一次 Decode 调用，让 past_key_values_decode 内部的 Cache 张量完成初始化。
        # 这一步不是最终导出，而是为后续捕获 Decode 图准备正确的 Cache 结构。
        model(
            input_ids=data_decode["input_ids"],
            past_key_values=past_key_values_decode,
            use_cache=True,
            cache_implementation="static",
        )

        # 捕获每次输入一个 token 的 Decode 路径。 得到计算图 graphs_decode
        graphs_decode = dynamo_compiler_decode.importer(
            model,
            input_ids=data_decode["input_ids"],
            use_cache=True,
            cache_position=cache_position, # 当前 token 在 Cache 中的位置。
            past_key_values=past_key_values_decode, # 已有的历史 KV Cache。
            cache_implementation="static",
        )
    else:
        # f32 和 bf16 使用相同的追踪流程，后面只在输出文件和参数存储上区分。
        past_key_values_prefill = StaticCache(
            config=model.config, max_cache_len=1024
        )
        past_key_values_decode = StaticCache(
            config=model.config, max_cache_len=1024
        )

        data_prefill = {
            "input_ids": torch.zeros((1, 1024), dtype=torch.int64),
        }
        data_decode = {
            "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        }

        cache_position = torch.tensor([200], dtype=torch.int64)

        # prefill 图使用完整 prompt 长度的输入张量。
        graphs_prefill = dynamo_compiler_prefill.importer(
            model,
            input_ids=data_prefill["input_ids"],
            use_cache=True,
            # past_key_values=past_key_values_prefill,
            cache_implementation="static",
        )
        # 在导入 decode 前先初始化一次 cache 结构。
        model(
            input_ids=data_decode["input_ids"],
            past_key_values=past_key_values_decode,
            use_cache=True,
            cache_implementation="static",
        )

        # decode 图表示逐 token 生成，并复用 KV cache。
        graphs_decode = dynamo_compiler_decode.importer(
            model,
            input_ids=data_decode["input_ids"],
            use_cache=True,
            cache_position=cache_position,
            past_key_values=past_key_values_decode,
            cache_implementation="static",
        )

if args.precision == "f16":
    # graphs_prefill、graphs_decode 是 DynamoCompiler.importer() 返回的计算图列表。
    # 两个 assert 要求 Prefill 和 Decode 各自只能捕获出一张完整计算图。
    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    # 检查通过后，分别取出列表中的第一张图，赋给 graph_prefill 和 graph_decode。
    graph_prefill = graphs_prefill[0]
    graph_decode = graphs_decode[0]

    # 从 Prefill 导入器中取出模型参数列表，包括权重和部分 buffer。参数顺序与计算图中的参数占位节点一致，后续会被展开并写入 arg0-f16.data。
    params = dynamo_compiler_prefill.imported_params[graph_prefill]
    # 依次对 Prefill 图执行两个原地变换：
    # eliminate_transpose：如果模型权重后面紧跟转置操作，就提前转置权重数据、修改参数形状并删除运行时转置节点。
    # 原计算：权重 → Transpose → MatMul
    # 优化后：预先转置的权重 → MatMul
    # eliminate_matmul_transpose_reshape：尝试消除 Transpose → Reshape/View 一类不改变有效数据布局的冗余组合。
    graphs_prefill[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    # 对 Decode 图执行相同优化。
    graphs_decode[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    # prefill 使用类 flash-attention 融合，decode 使用 GQA attention 融合。
    # Prefill 图的变换函数列表，执行顺序如下：
    # simply_fuse 把所有非 PlaceholderOp 的计算节点划分到一个 CPU 子图;
    # apply_classic_fusion 执行常规算子重写和融合
    # flash_attention_prefill 将 Prefill 图中的：ScaledDotProductFlashAttentionForCpuOp 替换为专用于长序列 Prefill 的： FlashAttentionForCpuPrefillOp
    pattern_list_prefill = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_list_decode = [
        simply_fuse,
        apply_classic_fusion,
        gqa_attention_fusion,
    ]
    # fuse_ops() 按列表顺序调用每个变换函数，并直接修改原 Buddy Graph.
    #每个列表的最后一个变换还会重新生成 subgraph0 分组，因此最终两张图都只有一个 CPU 计算子图，下一步再分别改名为 subgraph0_prefill 和 subgraph0_decode。
    graphs_prefill[0].fuse_ops(pattern_list_prefill)
    graphs_decode[0].fuse_ops(pattern_list_decode)

    # 重命名单个分组，避免 prefill 和 decode 生成的子图文件重名。
    graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
        "subgraph0"
    )
    # 本示例生成的 MLIR 面向 CPU 执行。
    graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

    graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
        "subgraph0"
    )
    graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

    # 创建 GraphDriver，用于构造子图模块和调用子图的 main graph。
    driver_prefill = GraphDriver(graphs_prefill[0])
    # 将 Prefill 子图中的 Buddy Op 转换成高层 MLIR 操作
    driver_prefill.subgraphs[0].lower_to_top_level_ir()

    driver_decode = GraphDriver(graphs_decode[0])
    driver_decode.subgraphs[0].lower_to_top_level_ir()
else:
    # f32/bf16 分支使用相同的图变换和 lowering 流程。
    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    graph_prefill = graphs_prefill[0]
    graph_decode = graphs_decode[0]

    params = dynamo_compiler_prefill.imported_params[graph_prefill]
    # 在融合前先消除 transpose 相关的冗余模式。
    graphs_prefill[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    graphs_decode[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    pattern_list_prefill = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_list_decode = [
        simply_fuse,
        apply_classic_fusion,
        gqa_attention_fusion,
    ]

    graphs_prefill[0].fuse_ops(pattern_list_prefill)
    graphs_decode[0].fuse_ops(pattern_list_decode)

    graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
        "subgraph0"
    )
    graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

    graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
        "subgraph0"
    )
    graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

    driver_prefill = GraphDriver(graphs_prefill[0])
    driver_prefill.subgraphs[0].lower_to_top_level_ir()

    driver_decode = GraphDriver(graphs_decode[0])
    driver_decode.subgraphs[0].lower_to_top_level_ir()

# 写出导入后的子图模块、main graph MLIR，以及打包后的权重数据。
if args.precision == "f16":
    # 把之前由 lower_to_top_level_ir() 生成的 Prefill MLIR 模块写入 subgraph0_prefill-f16.mlir
    with open(
        os.path.join(output_dir, "subgraph0_prefill-f16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    # 保存 Prefill 主图 到 forward_prefill-f16.mlir
    with open(
        os.path.join(output_dir, "forward_prefill-f16.mlir"), "w"
    ) as module_file:
        # do_param_pack=True 让 main graph 接口和打包后的参数保持一致。参数 True 表示启用参数打包
        print(driver_prefill.construct_main_graph(True), file=module_file)
    # 将所有参数张量展平成一段连续的二进制权重数据。
    # 对每个模型参数执行：detach()：脱离 PyTorch 自动求导系统。
    # numpy()：转换为 NumPy 数组。
    # reshape([-1])：将任意维度权重展平成一维。
    # numpy.concatenate()：按照计算图参数顺序拼成一段连续数组。
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    # 写入 arg0-f16.data 文件中
    all_param.tofile(os.path.join(output_dir, "arg0-f16.data"))

    # 保存 Decode 两个 MLIR 文件
    # subgraph0_decode-f16.mlir 中包含单 token Decode、GQA Attention 和 KV Cache 更新计算。
    with open(
        os.path.join(output_dir, "subgraph0_decode-f16.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode-f16.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
elif args.precision == "bf16":
    with open(
        os.path.join(output_dir, "subgraph0_prefill-bf16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_prefill-bf16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.construct_main_graph(True), file=module_file)
    # BF16 参数以 float32 数值的高 16 位形式存储。
    all_param = numpy.concatenate(
        [param.detach().float().numpy().reshape([-1]) for param in params]
    )
    all_param_bf16 = numpy.frombuffer(
        all_param.astype(numpy.float32).tobytes(), dtype=numpy.uint16
    )[1::2]
    all_param_bf16.tofile(os.path.join(output_dir, "arg0-bf16.data"))

    with open(
        os.path.join(output_dir, "subgraph0_decode-bf16.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode-bf16.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
else:
    with open(
        os.path.join(output_dir, "subgraph0_prefill.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_prefill.mlir"), "w"
    ) as module_file:
        print(driver_prefill.construct_main_graph(True), file=module_file)
    # 默认 f32 导出直接写出原始 float32 权重。
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))

    with open(
        os.path.join(output_dir, "subgraph0_decode.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
