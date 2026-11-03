"""The shared GPU route unwraps ReShade by replacing onnxruntime.dll's
delay-load import of D3D12CreateDevice (RedirectCreateDevice in
src/inference/ort_net.cc). That only works while the bundled runtime keeps
importing d3d12.dll through its delay-load table, so a runtime update that
links d3d12.dll directly or stops importing the function would silently leave
the wrapper in the model's way. Reads the DLL; nothing is loaded."""
import os
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
ORT = Path(os.environ.get('ORT', ROOT / 'deps/dml/microsoft.ml.onnxruntime.directml'))
DLL = ORT / 'runtimes/win-x64/native/onnxruntime.dll'
SDK_LIB = (Path(os.environ.get('BFA_MSVC_ROOT', ROOT / 'deps/commonlib/msvc'))
           / 'Windows Kits/10/Lib/10.0.26100.0/um/x64/d3d12.lib')
FUNCTION = b'D3D12CreateDevice'


class Image:
    """Enough of an x64 PE to read its import directories."""

    def __init__(self, blob):
        self.blob = blob
        if blob[:2] != b'MZ':
            raise ValueError('not a PE file')
        pe, = self.unpack('<I', 0x3c)
        if blob[pe:pe + 4] != b'PE\0\0':
            raise ValueError('invalid PE signature')
        machine, count = self.unpack('<HH', pe + 4)
        optional_size, = self.unpack('<H', pe + 20)
        self.optional = pe + 24
        if machine != 0x8664 or self.unpack('<H', self.optional)[0] != 0x20b:
            raise ValueError('expected x64 PE32+')
        if optional_size < 112 + 16 * 8:
            raise ValueError('missing data directories')
        self.sections = []
        for i in range(count):
            address, raw_size, raw = self.unpack('<III', self.optional + optional_size + i * 40 + 12)
            if raw + raw_size > len(blob):
                raise ValueError('truncated file-backed section')
            self.sections.append((address, raw_size, raw))

    def unpack(self, fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(self.blob):
            raise ValueError('truncated PE data')
        return struct.unpack_from(fmt, self.blob, offset)

    def offset(self, rva, size):
        for address, raw_size, raw in self.sections:
            if address <= rva and rva + size <= address + raw_size:
                return raw + rva - address
        raise ValueError('RVA outside file-backed section')

    def string(self, rva):
        start = self.offset(rva, 1)
        end = self.blob.find(b'\0', start, min(len(self.blob), start + 512))
        if end < 0:
            raise ValueError('unterminated name')
        return self.blob[start:end]

    def directory(self, index):
        return self.unpack('<II', self.optional + 112 + index * 8)

    def thunks(self, rva):
        """Import name table entries: ordinals as int, names as bytes."""
        entries = []
        for i in range(4096):
            value, = self.unpack('<Q', self.offset(rva + i * 8, 8))
            if not value:
                return entries
            entries.append(value & 0xffff if value >> 63 else self.string((value & 0xffffffff) + 2))
        raise ValueError('unreasonable import table')

    def imports(self):
        """Statically imported DLL names, lower-cased."""
        rva, _ = self.directory(1)
        names = set()
        for i in range(4096 if rva else 0):
            lookup, _, _, name, _ = self.unpack('<IIIII', self.offset(rva + i * 20, 20))
            if not name:
                break
            names.add(self.string(name).decode('ascii', 'replace').lower())
        return names

    def delay_imports(self):
        """Delay-loaded DLL name (lower-cased) to its import name table."""
        rva, _ = self.directory(13)
        table = {}
        for i in range(4096 if rva else 0):
            _, name, _, _, names, _, _, _ = self.unpack('<8I', self.offset(rva + i * 32, 32))
            if not name:
                break
            table[self.string(name).decode('ascii', 'replace').lower()] = self.thunks(names)
        return table


def sdk_import(lib, symbol):
    """How the Windows SDK import library binds symbol: its ordinal, or the
    name when d3d12.dll exports it by name."""
    blob = lib.read_bytes()
    if blob[:8] != b'!<arch>\n':
        raise ValueError('not an import library')
    at = 8
    while at + 60 <= len(blob):
        size = int(blob[at + 48:at + 58])
        member = blob[at + 60:at + 60 + size]
        if member[:4] == b'\0\0\xff\xff':
            hint, flags = struct.unpack_from('<HH', member, 16)
            if member[20:member.index(b'\0', 20)] == symbol:
                return hint if (flags >> 2) & 7 == 0 else symbol
        at += 60 + size + (size & 1)
    raise ValueError(f'{symbol.decode()} not in {lib}')


class DelayImportTest(unittest.TestCase):
    def test_runtime_delay_imports_create_device(self):
        self.assertTrue(DLL.is_file(), f'{DLL}: bundled runtime missing (ORT overrides the path)')
        self.assertTrue(SDK_LIB.is_file(), f'{SDK_LIB}: Windows SDK import library missing')
        image = Image(DLL.read_bytes())
        delayed = image.delay_imports()
        self.assertIn('d3d12.dll', delayed, 'onnxruntime.dll no longer delay-loads d3d12.dll')
        self.assertNotIn('d3d12.dll', image.imports(), 'a static d3d12.dll import bypasses the redirect')
        expected = sdk_import(SDK_LIB, FUNCTION)
        self.assertTrue(any(entry in (FUNCTION, expected) for entry in delayed['d3d12.dll']),
                        f'D3D12CreateDevice ({expected!r}) not among {delayed["d3d12.dll"]}')

    def test_damaged_runtime_is_rejected(self):
        blob = DLL.read_bytes()
        for size in (0, 63, 1024, len(blob) // 2):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    Image(blob[:size]).delay_imports()
        with self.assertRaises(ValueError):
            sdk_import(DLL, FUNCTION)


if __name__ == '__main__':
    unittest.main()
