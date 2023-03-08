# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import site
import subprocess
from pathlib import Path

import setuptools
import setuptools.command.build_ext


class make_ext(setuptools.command.build_ext.build_ext):  # type:ignore[misc]
    def build_extension(self, ext: setuptools.Extension) -> None:
        if ext.name == "libpoptorch_experimental_addons":
            filename = Path(self.build_lib) / self.get_ext_filename(
                self.get_ext_fullname(ext.name)
            )
            objdir = filename.with_suffix("")
            root_path = site.getsitepackages()[0]
            subprocess.check_call(
                [
                    "make",
                    f"OUT={filename}",
                    f"OBJDIR={objdir}",
                    f"ROOT_PATH={root_path}",
                ]
            )
        else:
            super().build_extension(ext)


setuptools.setup(
    name="poptorch-experimental-addons",
    version="0.1",
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
    ext_modules=[
        setuptools.Extension(
            "libpoptorch_experimental_addons",
            list(map(str, Path("poptorch_experimental_addons/cpp").glob("*.[ch]pp"))),
        )
    ],
    package_data={"poptorch_experimental_addons": ["cpp/*_codelet.cpp"]},
    cmdclass=dict(build_ext=make_ext),
)
