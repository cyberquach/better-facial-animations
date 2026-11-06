"""Minimal Skyrim SE (v105) BSA reader."""
import struct, lz4.frame

class BSA:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        magic, ver, off, self.flags, nfold, nfile, _, namelen, _, _ = struct.unpack('<4sIIIIIIIHH', self.f.read(36))
        assert magic == b'BSA\0' and ver == 105, (magic, ver)
        folders = [struct.unpack('<QIIQ', self.f.read(24)) for _ in range(nfold)]
        self.files = {}  # lowercase path -> (size, offset)
        recs = []
        for _, count, _, _ in folders:
            n = self.f.read(1)[0]
            name = self.f.read(n)[:-1].decode('cp1252')
            for _ in range(count):
                _, size, offset = struct.unpack('<QII', self.f.read(16))
                recs.append((name, size, offset))
        names = self.f.read(namelen)
        parts = names.split(b'\0')
        for (folder, size, offset), fn in zip(recs, parts):
            self.files[(folder + '\\' + fn.decode('cp1252')).lower()] = (size, offset)

    def read(self, path):
        size, offset = self.files[path.lower().replace('/', '\\')]
        compressed = bool(self.flags & 4) ^ bool(size & 0x40000000)
        size &= 0x3FFFFFFF
        self.f.seek(offset)
        if self.flags & 0x100:
            n = self.f.read(1)[0]; self.f.read(n); size -= n + 1
        if compressed:
            self.f.read(4)
            return lz4.frame.decompress(self.f.read(size - 4))
        return self.f.read(size)
