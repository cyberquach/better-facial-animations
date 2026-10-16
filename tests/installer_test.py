"""Resolve every installer combination to exactly one complete, consistent INI."""
import configparser
import importlib.util
import itertools
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('installer', ROOT / 'tools/package_installer.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory)
    installer.assemble(out, '4.3.0.7')
    # Vortex validates a UTF-16 copy; a retained UTF-8 declaration makes its
    # XmlReader fail before schema validation (Nexus-Mods/Vortex#17957).
    for name in ('ModuleConfig.xml', 'info.xml'):
        data = (out / 'fomod' / name).read_bytes()
        assert data.startswith(b'<config' if name == 'ModuleConfig.xml' else b'<fomod')
        assert b'<?xml' not in data
    assert ET.parse(out / 'fomod/info.xml').findtext('Version') == '4.3.0.7'
    config = ET.parse(out / 'fomod/ModuleConfig.xml')
    core = config.find('requiredInstallFiles/folder')
    assert core.attrib == {'source': 'SKSE', 'destination': 'SKSE', 'priority': '0'}
    steps = config.findall('installSteps/installStep')
    defaults = {}
    choices = []
    for group in config.findall('installSteps/installStep/optionalFileGroups/group'):
        plugins = group.findall('plugins/plugin')
        kind = group.attrib['type']
        if kind == 'SelectAll':
            continue
        plugin_flags = [{f.attrib['name']: f.text for f in p.findall('conditionFlags/flag')} for p in plugins]
        if kind == 'SelectExactlyOne':
            recommended = [i for i, p in enumerate(plugins) if p.find('typeDescriptor/type').attrib['name'] == 'Recommended']
            assert len(recommended) == 1
            defaults.update(plugin_flags[recommended[0]])
            choices.append(plugin_flags)
        else:
            assert kind == 'SelectAny'
            for plugin, flags in zip(plugins, plugin_flags):
                assert len(flags) == 1
                choices.append([{}, flags])
                if plugin.find('typeDescriptor/type').attrib['name'] == 'Recommended':
                    defaults.update(flags)
    assert defaults == {'GpuRoute': 'independent', 'Startup': 'gpu', 'ModelPreset': 'quality',
                        'PlayerAnimation': 'on', 'PrepareEarly': 'on'}
    patterns = config.findall('conditionalFileInstalls/patterns/pattern')
    assert len(patterns) == 96
    sources = set()
    for combination in itertools.product(*choices):
        flags = {key: value for choice in combination for key, value in choice.items()}
        matches = []
        for pattern in patterns:
            dependencies = pattern.find('dependencies')
            assert dependencies.attrib['operator'] == 'And'
            if all(flags.get(d.attrib['flag'], '') == d.attrib['value'] for d in dependencies):
                matches.extend(pattern.findall('files/file'))
        assert len(matches) == 1, flags
        file = matches[0]
        assert file.attrib['destination'] == 'SKSE\\Plugins\\BetterFacialAnimations.ini'
        assert int(file.attrib['priority']) > int(core.attrib['priority'])
        sources.add(file.attrib['source'])
        ini = configparser.ConfigParser()
        ini.read(out / file.attrib['source'].replace('\\', '/'))
        assert ini.getboolean('General', 'ForceCPU') == (flags['Startup'] == 'cpu')
        # CPU startup must preserve the route for subsequent live GPU opt-in.
        assert ini.getboolean('General', 'IndependentGpuDevice') == (flags['GpuRoute'] == 'independent')
        assert ini.getboolean('General', 'UsePerformanceModel') == (flags['ModelPreset'] == 'performance')
        assert ini.getboolean('General', 'PotatoMode') == (flags.get('ActiveOnly') == 'on')
        assert ini.getboolean('General', 'PlayerFacialAnimations') == (flags.get('PlayerAnimation') == 'on')
        assert ini.getboolean('General', 'Prefetch') == (flags.get('PrepareEarly') == 'on')
        assert ini.getint('General', 'SentimentAnalysis') == 0
        assert ini.getboolean('General', 'UseOldModel') == (flags['ModelPreset'] == 'old')
    assert len(sources) == 96
    raw = configparser.ConfigParser()
    raw.read(ROOT / 'dist/BetterFacialAnimations.ini')
    assert raw.getint('General', 'ForceCPU') == 0
    assert raw.getint('General', 'IndependentGpuDevice') == 1
    descriptions = [d.text for d in config.iter('description')]
    assert not any('@guide:' in d for d in descriptions)
    assert len(steps) == 4
    recovery = steps[-1].findall('optionalFileGroups/group/plugins/plugin')
    assert len(recovery) == 3
    for plugin in recovery:
        assert plugin.find('files') is None
        assert plugin.find('typeDescriptor/type').attrib['name'] == 'Required'
    assert 'ForceCPU=1' in recovery[0].findtext('description')
    assert 'CPU runs only when explicitly selected' in recovery[1].findtext('description')
    assert 'GPU-SETUP.txt' in recovery[2].findtext('description')
print('96 installer combinations, independent checkboxes, model selection, defaults and recovery: pass')
