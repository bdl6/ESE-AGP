"""支持 python -m ese_agp ...，把具体命令交给 cli.main 处理。"""
from .cli import main

# 仅作为命令运行时执行，其他模块导入此文件时不自动解析命令行。
if __name__ == "__main__":
    main()
