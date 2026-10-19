"""测试包。

存在这个文件（而非依赖隐式命名空间包）的目的是让 ``tests`` 成为一个真正的包，
从而使 ``from tests.conftest import ...`` 这类跨测试文件的辅助导入能够工作，
并保证 pytest 在 src layout 下把仓库根目录加入 sys.path。
"""
