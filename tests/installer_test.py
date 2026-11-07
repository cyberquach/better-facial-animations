"""Resolve every FOMOD selection and verify additive settings and package variants."""
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


def read_ini(path):
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(path)
    return {(section, key): value for section in ini.sections() for key, value in ini.items(section)}


def expected(flags):
    values = {
        ('General', 'ForceCPU'): flags['Startup'] == 'cpu',
        ('General', 'GpuAdapter'): 0 if flags['Startup'] == 'cpu' else int(flags['Startup'][-1]),
        ('General', 'EmotionDevice'): -1 if flags['EmotionDevice'] == 'cpu' else int(flags['EmotionDevice'][-1]),
        ('General', 'IndependentGpuDevice'): flags.get('GraphicsRoute') == 'independent',
        ('General', 'Enabled'): flags['FaceModel'] != 'disabled',
        ('General', 'UsePerformanceModel'): flags['FaceModel'] == 'performance',
        ('General', 'UseOldModel'): flags['FaceModel'] == 'old',
        ('General', 'SentimentAnalysis'): flags['EmotionModel'] != 'disabled',
        ('General', 'UsePerformanceEmotion'): flags['EmotionModel'] != 'full',
        ('LocalAnimationStrength', 'DisableTongue'): flags.get('Tongue') != 'on',
    }
    for flag, section, key in (
        ('VanillaEmotions', 'General', 'VanillaEmotions'),
        ('ActiveOnly', 'General', 'PotatoMode'),
        ('PlayerAnimation', 'General', 'PlayerFacialAnimations'),
        ('PrepareEarly', 'General', 'Prefetch'),
        ('Gaze', 'LocalAnimationStrength', 'ConversationGaze'),
        ('Logging', 'Debug', 'Logging'),
    ):
        values[section, key] = flags.get(flag) == 'on'
    for flag, ids in (('Khajiit', ('013745', '088845')),
                      ('Argonian', ('013740', '08883A'))):
        for form in ids:
            values['Races', 'Skyrim.esm|' + form] = flags.get(flag) == 'on'
    return {key: str(int(value)) for key, value in values.items()}


total = 0
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory)
    # Reuse one destination to expose stale Full choices/files on repackaging.
    for full_emotion in (True, False):
        installer.assemble(out, '5.0.0.6', full_emotion)
        for name in ('ModuleConfig.xml', 'info.xml'):
            data = (out / 'fomod' / name).read_bytes()
            assert data.startswith(b'<config' if name == 'ModuleConfig.xml' else b'<fomod')
            assert b'<?xml' not in data
        assert ET.parse(out / 'fomod/info.xml').findtext('Version') == '5.0.0.6'
        config = ET.parse(out / 'fomod/ModuleConfig.xml')
        core = config.find('requiredInstallFiles/folder')
        assert core.attrib == {'source': 'SKSE', 'destination': 'SKSE', 'priority': '0'}
        base = config.find('requiredInstallFiles/file')
        assert base.attrib['destination'] == 'SKSE\\Plugins\\BetterFacialAnimations.ini'
        assert int(base.attrib['priority']) > int(core.attrib['priority'])
        main = read_ini(out / base.attrib['source'].replace('\\', '/'))
        assert main['General', 'IniVersion'] == '9'
        defaults, choices = {}, []
        for group in config.findall('installSteps/installStep/optionalFileGroups/group'):
            plugins = group.findall('plugins/plugin')
            kind = group.attrib['type']
            if kind == 'SelectAll':
                assert all(p.find('files') is None for p in plugins)
                assert all(p.find('typeDescriptor/type').attrib['name'] == 'Required' for p in plugins)
                continue
            plugin_flags = [{f.attrib['name']: f.text for f in p.findall('conditionFlags/flag')} for p in plugins]
            assert all(len(flags) == 1 for flags in plugin_flags)
            if kind == 'SelectExactlyOne':
                recommended = [i for i, p in enumerate(plugins) if p.find('typeDescriptor/type').attrib['name'] == 'Recommended']
                assert len(recommended) == 1
                defaults.update(plugin_flags[recommended[0]])
                choices.append(plugin_flags)
            else:
                assert kind == 'SelectAny'
                for plugin, flags in zip(plugins, plugin_flags):
                    choices.append([{}, flags])
                    if plugin.find('typeDescriptor/type').attrib['name'] == 'Recommended':
                        defaults.update(flags)
        assert defaults == {
            'Startup': 'gpu0', 'FaceModel': 'quality',
            'EmotionDevice': 'gpu0',
            'EmotionModel': 'performance', 'VanillaEmotions': 'on',
            'PlayerAnimation': 'on', 'PrepareEarly': 'on', 'Tongue': 'on',
            'Gaze': 'on', 'Logging': 'on', 'Khajiit': 'on', 'Argonian': 'on',
        }
        # The recovery route is one unchecked checkbox, not a required choice.
        route_groups = [group for group in config.findall('.//group')
                        if group.find(".//flag[@name='GraphicsRoute']") is not None]
        assert len(route_groups) == 1
        route_group = route_groups[0]
        assert route_group.attrib['type'] == 'SelectAny'
        route_plugins = route_group.findall('plugins/plugin')
        assert len(route_plugins) == 1
        route_plugin = route_plugins[0]
        assert route_plugin.find('typeDescriptor/type').attrib['name'] == 'Optional'
        assert route_plugin.findtext("conditionFlags/flag[@name='GraphicsRoute']") == 'independent'
        patterns = config.findall('conditionalFileInstalls/patterns/pattern')
        assert len(patterns) == (37 if full_emotion else 36)
        lookup, ownership = {}, {}
        for pattern in patterns:
            dependencies = pattern.find('dependencies')
            assert dependencies.attrib['operator'] == 'And' and len(dependencies) == 1
            dep = dependencies[0]
            flag, value = dep.attrib['flag'], dep.attrib['value']
            file = pattern.find('files/file')
            destination = file.attrib['destination']
            assert destination == 'SKSE\\Plugins\\BetterFacialAnimations\\Installer\\' + flag + '.ini'
            assert (flag, value) not in lookup
            settings = read_ini(out / file.attrib['source'].replace('\\', '/'))
            assert not main.keys() & settings.keys(), (flag, 'main would hide installer choices')
            for key in settings:
                assert key not in ownership or ownership[key] == flag, (flag, key)
                ownership[key] = flag
            lookup[flag, value] = settings
        assert lookup['GraphicsRoute', ''] == {('General', 'IndependentGpuDevice'): '0'}
        assert lookup['GraphicsRoute', 'independent'] == {('General', 'IndependentGpuDevice'): '1'}
        assert ('EmotionModel', 'full') in lookup if full_emotion else ('EmotionModel', 'full') not in lookup
        assert (out / 'options/EmotionModel-full.ini').exists() == full_emotion
        # Both model selectors expose exactly the same independent hardware slots.
        device_choices = {'gpu0', 'gpu1', 'gpu2', 'gpu3', 'cpu'}
        for flag in ('Startup', 'EmotionDevice'):
            assert {value for key, value in lookup if key == flag} == device_choices
        source = (ROOT / 'src/plugin/settings.cc').read_text()
        for flag in ownership.values():
            assert '"' + flag + '"' in source, (flag, 'missing runtime loader')
        for combination in itertools.product(*choices):
            flags = {key: value for choice in combination for key, value in choice.items()}
            resolved = {}
            for flag in set(ownership.values()):
                resolved.update(lookup[flag, flags.get(flag, '')])
            assert resolved == expected(flags), flags
            total += 1
        manual = read_ini(ROOT / 'dist/BetterFacialAnimations.ini')
        for key, value in expected(defaults).items():
            if key[0] == 'Races':
                assert value == '1'  # built-in ordinary beast-race default
            else:
                assert manual[key] == value, key
        descriptions = [d.text for d in config.iter('description')]
        # The expanded per-OS recovery instructions are intentionally longer
        # than ordinary option descriptions (introduced with 5.0.4).
        route_description = route_plugin.findtext('description')
        assert route_description and len(route_description) < 1800
        assert all(section in route_description for section in
                   ('WINDOWS', 'LINUX / PROTON', 'GPU-SETUP.txt'))
        assert all('@guide:' not in text and
                   (text == route_description or len(text) < 800)
                   for text in descriptions)
        assert len(config.findall('installSteps/installStep')) == 6
        assert any('ForceCPU=1' in text and 'GPU-SETUP.txt' in text for text in descriptions)
        assert any('without SKSE Menu Framework' in text for text in descriptions)
        assert all('uncheck Force CPU' not in text for text in descriptions)
        assert 'Full — broader dialogue coverage' in (out / 'fomod/ModuleConfig.xml').read_text() if full_emotion else 'Full — broader dialogue coverage' not in (out / 'fomod/ModuleConfig.xml').read_text()
assert total == 512000
print(f'{total} installer combinations, disjoint defaults, both package variants and concise help: pass')
