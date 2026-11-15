"""The Performance model is described once: dist/performance-model.json. The
runtime's constants and every text a user reads state its size as that receipt
has it."""
import configparser
import importlib.util
import json
from pathlib import Path
import re
import unittest
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
# Each states the parameter count in millions.
TEXTS = ['README.md', 'dist/README.txt', 'dist/BetterFacialAnimations.ini',
         'dist/GPU-SETUP.txt', 'dist/fomod/ModuleConfig.xml',
         'src/plugin/settings.cc']
MILLIONS = re.compile(r'(\d+)(?:M|-million)[- ](?:parameter|Block AttnRes)')
WIDTH = re.compile(r'\(width (\d+)\)')


def read(name):
    return (ROOT / name).read_text().replace('"', '')


def constant(name):
    header = (ROOT / 'include/a2f/inference/live_core.h').read_text()
    value = re.search(rf'constexpr int {name} = ([\d\']+);', header)
    return int(value.group(1).replace("'", ''))


class ModelTextTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads(
            (ROOT / 'dist/performance-model.json').read_text())

    def test_runtime_constants(self):
        width = constant('kPerformanceWidth')
        heads = constant('kPerformanceHeads')
        self.assertEqual(constant('kPerformanceParameters'),
                         self.receipt['parameters'])
        self.assertEqual(
            self.receipt['architecture'],
            f"BlockAttnRes{width}-{constant('kPerformanceLayers')}layer-"
            f"KV{constant('kPerformanceMemoryFrames')}-block2")
        self.assertEqual(
            self.receipt['state']['shape'],
            [constant('kPerformanceLayers'), 1, heads,
             constant('kPerformanceMemoryFrames'), width // heads])

    def test_texts(self):
        millions = str(round(self.receipt['parameters'] / 1e6))
        width = str(constant('kPerformanceWidth'))
        for name in TEXTS:
            text = ' '.join((ROOT / name).read_text().replace('"', '').split())
            found = MILLIONS.findall(text)
            self.assertTrue(found, name)
            self.assertEqual(set(found), {millions}, name)
            self.assertLessEqual(set(WIDTH.findall(text)), {width}, name)

    def test_default_model(self):
        # The installer's recommended face model is the dist INI's, which is
        # the built-in default (sex_settings_test), and every text names it.
        spec = importlib.util.spec_from_file_location(
            'installer', ROOT / 'tools/package_installer.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        config = ElementTree.parse(ROOT / 'dist/fomod/ModuleConfig.xml')
        recommended = [
            plugin for plugin in config.iter('plugin')
            if plugin.find("conditionFlags/flag[@name='FaceModel']") is not None
            and plugin.find(".//type[@name='Recommended']") is not None]
        self.assertEqual(len(recommended), 1)
        flag = recommended[0].findtext(
            "conditionFlags/flag[@name='FaceModel']")
        name = recommended[0].get('name').split(' — ')[0]
        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ROOT / 'dist/BetterFacialAnimations.ini')
        choice = installer.CHOICES['FaceModel'][flag]['General']
        for key, value in choice.items():
            self.assertEqual(ini['General'][key], str(value), key)
        for path, phrase in (
                ('README.md', '{} facial animation and'),
                ('dist/README.txt', 'defaults to shared GPU, {} facial'),
                ('dist/README.txt', 'GPU-first {} is the normal preset'),
                ('dist/README.txt', 'by default {},'),
                ('dist/GPU-SETUP.txt',
                 'Shared GPU + {} is the initial preset')):
            text = ' '.join((ROOT / path).read_text().split())
            self.assertIn(phrase.format(name), text, path)

    def test_vram(self):
        # Every text states the face models' VRAM as net_keeper.h has it.
        header = (ROOT / 'include/a2f/inference/net_keeper.h').read_text()
        mb = {name: int(value) for name, value in re.findall(
            r'constexpr double kNetVramMb(\w*) = (\d+);', header)}
        v3 = (mb['3'], mb['3Fused'])
        perf = (mb['Performance'], mb['PerformanceFused'])
        for name in ('dist/BetterFacialAnimations.ini',
                     'dist/fomod/ModuleConfig.xml', 'src/plugin/settings.cc'):
            text = ' '.join(read(name).split())
            # V3 and Performance first; emotion figures may follow.
            found = re.findall(
                r'(?:about |~)(\d+) MB more(?: VRAM)? '
                r'\((\d+) (?:to|->) (\d+) MB\)', text)
            self.assertEqual(
                found[:2], [(str(on - off), str(off), str(on))
                        for off, on in (v3, perf)], name)
        text = ' '.join(read('dist/GPU-SETUP.txt').split())
        self.assertIn(f'Fusion adds about {v3[1] - v3[0]} MB for V3 ', text)
        self.assertIn(f', {perf[1] - perf[0]} MB for Performance ', text)
        text = ' '.join(read('CHANGELOG.md').split())
        self.assertIn(f'V3 drops from about {v3[1]} to {v3[0]} MB, '
                      f'Performance from {perf[1]} to {perf[0]} MB', text)
        for name in ('README.md', 'dist/BetterFacialAnimations.ini',
                     'dist/GPU-SETUP.txt', 'dist/fomod/ModuleConfig.xml',
                     'src/plugin/settings.cc'):
            text = ' '.join(read(name).lower().split())
            for off in (v3[0], perf[0]):
                self.assertTrue(f'about {off} mb' in text, f'{name}: {off}')

    def test_sees_another_size(self):
        self.assertEqual(
            MILLIONS.findall('Performance (4M Block AttnRes) runs on GPU'),
            ['4'])
        self.assertEqual(MILLIONS.findall('a 4-million-parameter student'),
                         ['4'])


if __name__ == '__main__':
    unittest.main()
