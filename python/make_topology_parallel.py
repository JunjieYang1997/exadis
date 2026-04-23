# This script is used to generate cpp files that instantiate the
# make_topology_parallel function with template arguments passed 
# from CMake to reduce compilation times with GPU compilers
# Nicolas Bertin
#功能概述：本脚本用于自动生成 C++ 源文件和头文件，将 make_topology_parallel 模板函数
#的不同实例化拆分到独立的编译单元中，从而实现并行编译，减少 GPU 编译器(如 NVCC)的编译时间。
# 使用方式（由 CMake 构建系统调用）：
#   1. 生成头文件（传入多个参数）：
#      python make_topology_parallel.py TypeA TypeB TypeC ...
#      -> 生成 topology_parallel_types.h，包含所有类型的 extern template 声明
#
#   2. 生成单个源文件（传入一个参数）：
#      python make_topology_parallel.py TypeA
#      -> 生成 topology_TypeA.cpp，包含 TypeA 对应的显式模板实例化
#
# 作者：Nicolas Bertin
import sys# 获取命令行参数

args = sys.argv[1:]

    # =========================================================================
    # 模式一：传入多个参数 —— 生成全局头文件
    # =========================================================================
    # 当传入多个力类型参数时，生成一个公共头文件 topology_parallel_types.h。
    # 该头文件包含所有力类型对应的 extern template 声明。
    #
    # extern template 的作用：
    #   告诉编译器"该模板实例化将在其他编译单元中完成"，
    #   防止编译器在包含此头文件的每个 .cpp 文件中重复实例化同一模板，
    #   从而避免冗余编译，减少编译时间和目标文件体积。
    # =========================================================================
if len(args) > 1:
    # generate global header file
    filename = 'topology_parallel_types.h'
    with open(filename, 'w') as f:
        for arg in args:
            f.write('extern template Topology* make_topology_parallel<ForceType::%s>(System* system, Force* force, Mobility* mobility, TParams& topolparams);\n' % arg)
    # =========================================================================
    # 模式二：传入单个参数 —— 生成独立的 .cpp 源文件
    # =========================================================================
    # 当只传入一个力类型参数时，生成该类型对应的独立 .cpp 文件。
    # 每个文件只包含一个显式模板实例化（explicit template instantiation），
    # 使得各个类型可以在不同的编译单元中独立、并行地编译。
    #
    # 显式模板实例化的作用：
    #   强制编译器在当前编译单元中生成该模板特化的完整代码，
    #   与头文件中的 extern template 声明配合使用，
    #   确保整个项目中每种类型只被实例化一次。
    # =========================================================================        
else:
    # generate individual cpp files
    arg = args[0]
    filename = 'topology_%s.cpp' % arg
    with open(filename, 'w') as f:
        f.write('#include "exadis_pybind.h"\n')
        f.write('template Topology* make_topology_parallel<ForceType::%s>(System* system, Force* force, Mobility* mobility, TParams& topolparams);\n' % arg)
