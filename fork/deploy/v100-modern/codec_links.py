"""Expose PyAV's pinned FFmpeg SONAMEs to the matching TorchCodec loader.

PyAV wheels give their bundled libraries hashed filenames. TorchCodec expects
ordinary FFmpeg SONAMEs. Relative aliases in that same directory preserve the
bundled dependencies and avoid shipping an older second FFmpeg distribution.
"""

import json
import sysconfig
from pathlib import Path


def main():
    root = Path(sysconfig.get_paths()["purelib"]) / "av.libs"
    aliases = {}
    for name in [
        "avcodec",
        "avformat",
        "avutil",
        "avdevice",
        "avfilter",
        "swscale",
        "swresample",
    ]:
        files = list(root.glob(f"lib{name}-*.so.*"))
        assert len(files) == 1, (name, files)
        source = files[0]
        major = source.name.split(".so.", 1)[1].split(".", 1)[0]
        alias = root / f"lib{name}.so.{major}"
        if alias.exists():
            assert alias.resolve() == source.resolve()
        else:
            alias.symlink_to(source.name)
        aliases[alias.name] = source.name
    print(json.dumps(aliases, sort_keys=True))


if __name__ == "__main__":
    main()
