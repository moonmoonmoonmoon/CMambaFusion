import numpy as np
import pickle


class SceneElements(object):
    def __init__(self, rules, **kwargs):
        super().__init__()
        self.rules = rules
        self.values_map_dict = self.init_values_map(rules)

    def init_values_map(self, rules):
        values_map_dict = {}
        for elem, elem_v in rules.items():
            values, unique = elem_v['values'], elem_v['unique']
            for v in values:
                values_map_dict[v] = [elem, unique]
        return values_map_dict

    def get_template_elemets(self):
        elem_dict = {}
        for elem, elem_v in self.rules.items():
            values, unique = elem_v['values'], elem_v['unique']
            if unique:
                elem_dict[elem] = ""
            else:
                elem_dict[elem] = []
        return elem_dict

    def get_elements(self, scene):
        elem_dict = self.get_template_elemets()
        scene_split = scene.split('.')
        for scene_value in scene_split:
            assert scene_value in self.values_map_dict, f'SceneElements Error: element - {scene_value} not in scene rules.'
            elem, unique = self.values_map_dict[scene_value]
            if unique:
                elem_dict[elem] = scene_value
            else:
                elem_dict[elem].append(scene_value)
        return elem_dict


seg_learning_map_9 = {
    255: 0,  # unlabeled
    10: 1,  # drivable surface
    11: 2,  # curb
    30: 3,  # other ground
    40: 4,  # car
    60: 5,  # ride
    80: 6,  # person
    90: 7,  # moving obstacle in the drivable surface
    100: 7,  # static obstacle in the drivable surface
    120: 8,  # moving obstacle outside the drivable surface
    130: 8,  # static obstacle outsize the drivable surface
    250: 0,  # ego
    251: 9,  # noise
}

seg_learning_map_2 = {
    255: 0,  # unlabeled
    10: 1,  # drivable surface
    11: 2,  # curb
    30: 0,  # other ground
    40: 0,  # car
    60: 0,  # ride
    80: 0,  # person
    90: 0,  # moving obstacle in the drivable surface
    100: 0,  # static obstacle in the drivable surface
    120: 0,  # moving obstacle outside the drivable surface
    130: 0,  # static obstacle outsize the drivable surface
    250: 0,  # ego
    251: 0,  # noise
}

seg_learning_map_3 = {
    255: 0,  # unlabeled
    10: 1,  # drivable surface
    11: 2,  # curb
    30: 3,  # other ground
    40: 3,  # car
    60: 3,  # ride
    80: 3,  # person
    90: 3,  # moving obstacle in the drivable surface
    100: 3,  # static obstacle in the drivable surface
    120: 3,  # moving obstacle outside the drivable surface
    130: 3,  # static obstacle outsize the drivable surface
    250: 3,  # ego
    251: 3,  # noise
}

seg_learning_map_8 = {
    255: 0,    # unlabeled
    10: 1,    # drivable surface
    11: 2,    # curb
    30: 3,    # other ground
    40: 4,    # car
    60: 5,    # ride
    80: 5,    # person
    90: 6,    # moving obstacle in the drivable surface
    100: 6,    # static obstacle in the drivable surface
    120: 7,    # moving obstacle outside the drivable surface
    130: 7,    # static obstacle outsize the drivable surface
    250: 0,    # ego
    251: 8,    # noise
}

seg_learning_map_pred = {
      0: 0,     # unknown
     10: 1,     # ground
     11: 2,     # roadedge
     30: 3,     # walkable_ground
     40: 4,     # car
     60: 5,     # VRU
    100: 6,     # cone
    130: 7,     # background
    250: 8,     # noise
}

seg_learning_map = seg_learning_map_9

seg_inverse_learning_map = {
    0: 255,  # unlabeled
    1: 10,  # drivable surface
    2: 11,  # curb
    3: 30,  # other ground
    4: 40,  # car
    5: 60,  # ride
    6: 80,  # person
    7: 90,  # obstacle in the drivable surface
    8: 120,  # obstacle outside the drivable surface
    # 9: 250, # ego
    9: 251,  # noise
}

seg_class_name = [
    'unlabeled', 'drivable-surface', 'curb', 'other-ground', 'car', 'ride',
    'person', 'obstacle-in', 'obstacle-out', 'noise',
]

# seg_class_name = [
#     'unknown', 'drivable-surface', 'curb', 'other-ground', 'car', 'VRU',
#     'cone', 'background', 'noise',
# ]
