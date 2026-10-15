from pathlib import Path
import zipfile


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/basic_flight'


def archive(directory, destination):
    with zipfile.ZipFile(destination, 'w') as stream:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                stream.write(path, path.relative_to(directory))
    return destination


def add_fixed_anchors(request, index=None):
    """CPU/GPU均使用两个不同朝向的固定端点，其中一个与飞行器起点重叠。"""
    group = dict(group_id='anchor_group', model_id='radio_base_station', model_revision=1,
                 count=2, placement_id='anchor_start', controller_id='none', sensor_profile_id='none')
    groups = request['scenario']['entity_groups']
    groups.insert(len(groups) if index is None else index, group)
    request['scenario']['placements']['anchor_start'] = [
        dict(position=[0., 1., 1.], orientation_wxyz=[1., 0., 0., 0.]),
        dict(position=[3., 2., 1.5], orientation_wxyz=[0., 0., 0., 1.]),
    ]
