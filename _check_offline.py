import src.runner.offline_run as m
import inspect
src = inspect.getsource(m.build_parser)
with open("_check_out.txt", "w") as f:
    f.write(src)
print("written")
