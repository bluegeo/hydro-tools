from types import SimpleNamespace
import os


class TestFiles(SimpleNamespace):
    def __init__(self):
        d = os.path.join(os.path.dirname(__file__), "data-files")
        super().__init__(
            **{".".join(f.split(".")[:-1]): os.path.join(d, f) for f in os.listdir(d)}
        )


test_data = TestFiles()
