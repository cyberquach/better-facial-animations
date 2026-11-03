"""Read user-supplied Skyrim BSA archives without extracting member paths."""
from pathlib import Path
import struct
import zlib
import lz4.frame


class BSA:
    """Read v104/v105 archive indexes; open payloads only when requested."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            magic, version, offset, flags, folders, count, _, names_size, _ = struct.unpack('<4s8I', f.read(36))
            if magic != b'BSA\0' or version not in (104, 105):
                raise ValueError(f'unsupported BSA {magic!r} / {version}')
            self.version, self.flags = version, flags
            f.seek(offset)
            fmt = '<QIIQ' if version == 105 else '<QII'
            entries = [struct.unpack(fmt, f.read(struct.calcsize(fmt))) for _ in range(folders)]
            records = []
            for entry in entries:
                n = f.read(1)[0]
                folder = f.read(n).rstrip(b'\0').decode('cp1252')
                for _ in range(entry[1]):
                    _, size, position = struct.unpack('<QII', f.read(16))
                    records.append((folder, size, position))
            names = f.read(names_size).split(b'\0')
            if len(records) != count or len(names) < count:
                raise ValueError('incomplete BSA index')
        self.files = {}
        for (folder, size, position), name in zip(records, names):
            member = (folder + '/' + name.decode('cp1252')).replace('\\', '/').lower()
            self.files[member] = (size, position)

    def read(self, member):
        size, offset = self.files[member]
        compressed = bool(self.flags & 4) ^ bool(size & 0x40000000)
        size &= 0x3fffffff
        with self.path.open('rb') as f:
            f.seek(offset)
            if self.flags & 0x100:
                n = f.read(1)[0]
                f.read(n)
                size -= n + 1
            raw = f.read(size)
        if len(raw) != size:
            raise ValueError('truncated BSA payload')
        if compressed:
            expected = struct.unpack_from('<I', raw)[0]
            raw = lz4.frame.decompress(raw[4:]) if self.version == 105 else zlib.decompress(raw[4:])
            if len(raw) != expected:
                raise ValueError('BSA decompressed size mismatch')
        return raw
