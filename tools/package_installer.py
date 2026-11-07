#!/usr/bin/env python3
"""Assemble FOMOD metadata and independent, additive installer defaults.

The main INI wins over these defaults, so later menu or manual edits persist.
No helper executable is packaged or launched by the mod.
"""
import argparse
import configparser
import io
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEVICES = {'gpu0': 0, 'gpu1': 1, 'gpu2': 2, 'gpu3': 3, 'cpu': -1}
# Each choice owns distinct keys. Fixed destination names replace prior choices
# on reinstall; no exponential collection of complete INIs is needed.
CHOICES = {
    'GraphicsRoute': {
        '': {'General': {'IndependentGpuDevice': 0}},
        'independent': {'General': {'IndependentGpuDevice': 1}},
    },
    'Startup': {
        value: {'General': {'ForceCPU': int(device == -1),
                            'GpuAdapter': max(0, device)}}
        for value, device in DEVICES.items()
    },
    'EmotionDevice': {
        value: {'General': {'EmotionDevice': device}}
        for value, device in DEVICES.items()
    },
    'FaceModel': {
        value: {'General': {'Enabled': int(value != 'disabled'),
                            'UsePerformanceModel': int(value == 'performance'),
                            'UseUltraPerformanceModel': int(value == 'ultra'),
                            'UseOldModel': int(value == 'old')}}
        for value in ('quality', 'performance', 'ultra', 'old', 'disabled')
    },
    'EmotionModel': {
        value: {'General': {'SentimentAnalysis': int(value != 'disabled'),
                            'UsePerformanceEmotion': int(value != 'full')}}
        for value in ('performance', 'full', 'disabled')
    },
}
for flag, section, key, inverted in (
    ('VanillaEmotions', 'General', 'VanillaEmotions', False),
    ('ActiveOnly', 'General', 'PotatoMode', False),
    ('PlayerAnimation', 'General', 'PlayerFacialAnimations', False),
    ('PrepareEarly', 'General', 'Prefetch', False),
    ('Tongue', 'LocalAnimationStrength', 'DisableTongue', True),
    ('Gaze', 'LocalAnimationStrength', 'ConversationGaze', False),
    ('Logging', 'Debug', 'Logging', False),
):
    CHOICES[flag] = {value: {section: {key: int((value == 'on') != inverted)}}
                     for value in ('', 'on')}
for flag, ids in (('Khajiit', ('013745', '088845')),
                  ('Argonian', ('013740', '08883A'))):
    CHOICES[flag] = {
        value: {'Races': {'Skyrim.esm|' + form: int(value == 'on') for form in ids}}
        for value in ('', 'on')
    }


def serialize(settings):
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read_dict(settings)
    out = io.StringIO()
    ini.write(out, space_around_delimiters=False)
    return out.getvalue()


def assemble(out, version, full_emotion=True):
    (out / 'fomod').mkdir(parents=True, exist_ok=True)
    (out / 'options').mkdir(exist_ok=True)
    config = ET.parse(ROOT / 'dist/fomod/ModuleConfig.xml')
    # A missing model cannot be selected, even if an older install had it.
    if not full_emotion:
        for plugins in config.findall('.//plugins'):
            for plugin in list(plugins):
                if plugin.findtext("conditionFlags/flag[@name='EmotionModel']") == 'full':
                    plugins.remove(plugin)
    patterns = config.find('conditionalFileInstalls/patterns')
    patterns.clear()
    for stale in (out / 'options').glob('*.ini'):
        stale.unlink()
    controlled = set()
    for flag, choices in CHOICES.items():
        for value, settings in choices.items():
            if flag == 'EmotionModel' and value == 'full' and not full_emotion:
                continue
            name = flag + '-' + (value or 'off') + '.ini'
            (out / 'options' / name).write_text(serialize(settings), encoding='utf-8')
            controlled.update((section, key) for section, keys in settings.items() for key in keys)
            pattern = ET.SubElement(patterns, 'pattern')
            dependencies = ET.SubElement(pattern, 'dependencies', operator='And')
            ET.SubElement(dependencies, 'flagDependency', flag=flag, value=value)
            ET.SubElement(ET.SubElement(pattern, 'files'), 'file',
                          source='options\\' + name,
                          destination='SKSE\\Plugins\\BetterFacialAnimations\\Installer\\' + flag + '.ini',
                          priority='10')
    # Keep comments and fine controls; installer-owned keys must be absent so
    # their fragments supply the initial value. Manual installations still use
    # the original complete dist INI, with identical built-in defaults.
    section = ''
    lines = ['; Installer choices are defaults; explicit values here win.\n']
    for line in (ROOT / 'dist/BetterFacialAnimations.ini').read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1]
        key = stripped.partition('=')[0]
        if (section, key) in controlled:
            lines.append('; ' + line)
        else:
            lines.append(line)
    (out / 'options/Settings.ini').write_text(''.join(lines), encoding='utf-8')
    ET.SubElement(config.find('requiredInstallFiles'), 'file',
                  source='options\\Settings.ini',
                  destination='SKSE\\Plugins\\BetterFacialAnimations.ini', priority='10')
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.indent(config, space='  ')
    # Vortex re-encodes validation input as UTF-16 but only strips double-quoted
    # encoding attributes. Omit the declaration to avoid its encoding mismatch.
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
    parser.add_argument('--without-full-emotion', action='store_true')
    args = parser.parse_args()
    assemble(args.output, args.version, not args.without_full_emotion)
