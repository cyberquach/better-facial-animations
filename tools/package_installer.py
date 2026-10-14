#!/usr/bin/env python3
"""Assemble FOMOD metadata and complete INI presets beside the installable mod.

The caller archives SKSE from build/mod and fomod/options from this directory.
No helper executable is packaged or launched by the mod.
"""
import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    f'{route}-{startup}-{model}-{active}{player}{early}': {
        'ForceCPU': int(startup == 'cpu'),
        'IndependentGpuDevice': int(route == 'independent'),
        'UsePerformanceModel': int(model == 'performance'),
        'UseOldModel': int(model == 'old'),
        'PotatoMode': active,
        'PlayerFacialAnimations': player,
        'Prefetch': early,
    }
    for route in ('independent', 'shared')
    for startup in ('gpu', 'cpu')
    for model in ('quality', 'performance', 'old')
    for active in (0, 1)
    for player in (0, 1)
    for early in (0, 1)
}


def assemble(out, version):
    (out / 'fomod').mkdir(parents=True, exist_ok=True)
    (out / 'options').mkdir(exist_ok=True)
    template = (ROOT / 'dist/BetterFacialAnimations.ini').read_text()
    config = ET.parse(ROOT / 'dist/fomod/ModuleConfig.xml')
    patterns = config.find('conditionalFileInstalls/patterns')
    patterns.clear()
    for stale in (out / 'options').glob('*.ini'):
        stale.unlink()
    for name, settings in PRESETS.items():
        contents = template
        for key, value in {**settings, 'SentimentAnalysis': 0}.items():
            contents, count = re.subn(r'^' + key + r'=.*$', f'{key}={value}',
                                     contents, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f'{key}: expected exactly one setting, got {count}')
        (out / 'options' / (name + '.ini')).write_text(contents)
        route, startup, model, checks = name.split('-')
        pattern = ET.SubElement(patterns, 'pattern')
        dependencies = ET.SubElement(pattern, 'dependencies', operator='And')
        flags = {'GpuRoute': route, 'Startup': startup, 'ModelPreset': model}
        # An unchecked SelectAny option leaves its flag unset (empty value).
        flags.update(zip(('ActiveOnly', 'PlayerAnimation', 'PrepareEarly'),
                         ('on' if v == '1' else '' for v in checks)))
        for flag, value in flags.items():
            ET.SubElement(dependencies, 'flagDependency', flag=flag, value=value)
        ET.SubElement(ET.SubElement(pattern, 'files'), 'file',
                      source='options\\' + name + '.ini',
                      destination='SKSE\\Plugins\\BetterFacialAnimations.ini',
                      priority='10')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.indent(config, space='  ')
    # Vortex re-encodes validation input as UTF-16 but only strips double-quoted
    # encoding attributes. ElementTree's single-quoted declaration breaks it.
    config.write(out / 'fomod/ModuleConfig.xml', encoding='utf-8', xml_declaration=False)
    info = ET.Element('fomod')
    for key, value in [('Name', 'Better Facial Animations'), ('Version', version),
                       ('Website', 'https://www.nexusmods.com/skyrimspecialedition/mods/193598')]:
        ET.SubElement(info, key).text = value
    ET.ElementTree(info).write(out / 'fomod/info.xml', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('version')
    args = parser.parse_args()
    assemble(args.output, args.version)
