import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text


class VOT2021_LTDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot2019_lt_path
        self.sequence_info_list = self._get_sequence_info_list()
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])
    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]


        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        return Sequence(sequence_info['name'], frames, 'vot2021_lt', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list=[{"name": "ballet", "path": "sequences/ballet", "startFrame": 1, "endFrame": 1389, "nz": 8, "ext": "jpg", "anno_path": "annotations/ballet/groundtruth.txt", "object_class": "ballet"}, {"name": "bicycle", "path": "sequences/bicycle", "startFrame": 1, "endFrame": 2842, "nz": 8, "ext": "jpg", "anno_path": "annotations/bicycle/groundtruth.txt", "object_class": "bicycle"}, {"name": "bike1", "path": "sequences/bike1", "startFrame": 1, "endFrame": 3085, "nz": 8, "ext": "jpg", "anno_path": "annotations/bike1/groundtruth.txt", "object_class": "bike1"}, {"name": "bird1", "path": "sequences/bird1", "startFrame": 1, "endFrame": 2437, "nz": 8, "ext": "jpg", "anno_path": "annotations/bird1/groundtruth.txt", "object_class": "bird1"}, {"name": "boat", "path": "sequences/boat", "startFrame": 1, "endFrame": 7524, "nz": 8, "ext": "jpg", "anno_path": "annotations/boat/groundtruth.txt", "object_class": "boat"}, {"name": "bull", "path": "sequences/bull", "startFrame": 1, "endFrame": 1907, "nz": 8, "ext": "jpg", "anno_path": "annotations/bull/groundtruth.txt", "object_class": "bull"}, {"name": "car1", "path": "sequences/car1", "startFrame": 1, "endFrame": 2629, "nz": 8, "ext": "jpg", "anno_path": "annotations/car1/groundtruth.txt", "object_class": "car1"}, {"name": "car16", "path": "sequences/car16", "startFrame": 1, "endFrame": 1993, "nz": 8, "ext": "jpg", "anno_path": "annotations/car16/groundtruth.txt", "object_class": "car16"}, {"name": "car3", "path": "sequences/car3", "startFrame": 1, "endFrame": 1717, "nz": 8, "ext": "jpg", "anno_path": "annotations/car3/groundtruth.txt", "object_class": "car3"}, {"name": "car6", "path": "sequences/car6", "startFrame": 1, "endFrame": 4861, "nz": 8, "ext": "jpg", "anno_path": "annotations/car6/groundtruth.txt", "object_class": "car6"}, {"name": "car8", "path": "sequences/car8", "startFrame": 1, "endFrame": 2575, "nz": 8, "ext": "jpg", "anno_path": "annotations/car8/groundtruth.txt", "object_class": "car8"}, {"name": "car9", "path": "sequences/car9", "startFrame": 1, "endFrame": 1879, "nz": 8, "ext": "jpg", "anno_path": "annotations/car9/groundtruth.txt", "object_class": "car9"}, {"name": "carchase", "path": "sequences/carchase", "startFrame": 1, "endFrame": 9928, "nz": 8, "ext": "jpg", "anno_path": "annotations/carchase/groundtruth.txt", "object_class": "carchase"}, {"name": "cat1", "path": "sequences/cat1", "startFrame": 1, "endFrame": 1664, "nz": 8, "ext": "jpg", "anno_path": "annotations/cat1/groundtruth.txt", "object_class": "cat1"}, {"name": "cat2", "path": "sequences/cat2", "startFrame": 1, "endFrame": 2756, "nz": 8, "ext": "jpg", "anno_path": "annotations/cat2/groundtruth.txt", "object_class": "cat2"}, {"name": "deer", "path": "sequences/deer", "startFrame": 1, "endFrame": 1500, "nz": 8, "ext": "jpg", "anno_path": "annotations/deer/groundtruth.txt", "object_class": "deer"}, {"name": "dog", "path": "sequences/dog", "startFrame": 1, "endFrame": 1829, "nz": 8, "ext": "jpg", "anno_path": "annotations/dog/groundtruth.txt", "object_class": "dog"}, {"name": "dragon", "path": "sequences/dragon", "startFrame": 1, "endFrame": 4474, "nz": 8, "ext": "jpg", "anno_path": "annotations/dragon/groundtruth.txt", "object_class": "dragon"}, {"name": "f1", "path": "sequences/f1", "startFrame": 1, "endFrame": 3535, "nz": 8, "ext": "jpg", "anno_path": "annotations/f1/groundtruth.txt", "object_class": "f1"}, {"name": "following", "path": "sequences/following", "startFrame": 1, "endFrame": 12509, "nz": 8, "ext": "jpg", "anno_path": "annotations/following/groundtruth.txt", "object_class": "following"}, {"name": "freesbiedog", "path": "sequences/freesbiedog", "startFrame": 1, "endFrame": 3176, "nz": 8, "ext": "jpg", "anno_path": "annotations/freesbiedog/groundtruth.txt", "object_class": "freesbiedog"}, {"name": "freestyle", "path": "sequences/freestyle", "startFrame": 1, "endFrame": 2256, "nz": 8, "ext": "jpg", "anno_path": "annotations/freestyle/groundtruth.txt", "object_class": "freestyle"}, {"name": "group1", "path": "sequences/group1", "startFrame": 1, "endFrame": 4873, "nz": 8, "ext": "jpg", "anno_path": "annotations/group1/groundtruth.txt", "object_class": "group1"}, {"name": "group2", "path": "sequences/group2", "startFrame": 1, "endFrame": 2683, "nz": 8, "ext": "jpg", "anno_path": "annotations/group2/groundtruth.txt", "object_class": "group2"}, {"name": "group3", "path": "sequences/group3", "startFrame": 1, "endFrame": 5527, "nz": 8, "ext": "jpg", "anno_path": "annotations/group3/groundtruth.txt", "object_class": "group3"}, {"name": "helicopter", "path": "sequences/helicopter", "startFrame": 1, "endFrame": 4519, "nz": 8, "ext": "jpg", "anno_path": "annotations/helicopter/groundtruth.txt", "object_class": "helicopter"}, {"name": "horseride", "path": "sequences/horseride", "startFrame": 1, "endFrame": 14485, "nz": 8, "ext": "jpg", "anno_path": "annotations/horseride/groundtruth.txt", "object_class": "horseride"}, {"name": "kitesurfing", "path": "sequences/kitesurfing", "startFrame": 1, "endFrame": 4665, "nz": 8, "ext": "jpg", "anno_path": "annotations/kitesurfing/groundtruth.txt", "object_class": "kitesurfing"}, {"name": "liverRun", "path": "sequences/liverRun", "startFrame": 1, "endFrame": 29700, "nz": 8, "ext": "jpg", "anno_path": "annotations/liverRun/groundtruth.txt", "object_class": "liverRun"}, {"name": "longboard", "path": "sequences/longboard", "startFrame": 1, "endFrame": 7059, "nz": 8, "ext": "jpg", "anno_path": "annotations/longboard/groundtruth.txt", "object_class": "longboard"}, {"name": "nissan", "path": "sequences/nissan", "startFrame": 1, "endFrame": 3800, "nz": 8, "ext": "jpg", "anno_path": "annotations/nissan/groundtruth.txt", "object_class": "nissan"}, {"name": "parachute", "path": "sequences/parachute", "startFrame": 1, "endFrame": 2858, "nz": 8, "ext": "jpg", "anno_path": "annotations/parachute/groundtruth.txt", "object_class": "parachute"}, {"name": "person14", "path": "sequences/person14", "startFrame": 1, "endFrame": 2923, "nz": 8, "ext": "jpg", "anno_path": "annotations/person14/groundtruth.txt", "object_class": "person14"}, {"name": "person17", "path": "sequences/person17", "startFrame": 1, "endFrame": 2347, "nz": 8, "ext": "jpg", "anno_path": "annotations/person17/groundtruth.txt", "object_class": "person17"}, {"name": "person19", "path": "sequences/person19", "startFrame": 1, "endFrame": 4357, "nz": 8, "ext": "jpg", "anno_path": "annotations/person19/groundtruth.txt", "object_class": "person19"}, {"name": "person2", "path": "sequences/person2", "startFrame": 1, "endFrame": 2623, "nz": 8, "ext": "jpg", "anno_path": "annotations/person2/groundtruth.txt", "object_class": "person2"}, {"name": "person20", "path": "sequences/person20", "startFrame": 1, "endFrame": 1783, "nz": 8, "ext": "jpg", "anno_path": "annotations/person20/groundtruth.txt", "object_class": "person20"}, {"name": "person4", "path": "sequences/person4", "startFrame": 1, "endFrame": 2743, "nz": 8, "ext": "jpg", "anno_path": "annotations/person4/groundtruth.txt", "object_class": "person4"}, {"name": "person5", "path": "sequences/person5", "startFrame": 1, "endFrame": 2101, "nz": 8, "ext": "jpg", "anno_path": "annotations/person5/groundtruth.txt", "object_class": "person5"}, {"name": "person7", "path": "sequences/person7", "startFrame": 1, "endFrame": 2065, "nz": 8, "ext": "jpg", "anno_path": "annotations/person7/groundtruth.txt", "object_class": "person7"}, {"name": "rollerman", "path": "sequences/rollerman", "startFrame": 1, "endFrame": 1712, "nz": 8, "ext": "jpg", "anno_path": "annotations/rollerman/groundtruth.txt", "object_class": "rollerman"}, {"name": "sitcom", "path": "sequences/sitcom", "startFrame": 1, "endFrame": 3898, "nz": 8, "ext": "jpg", "anno_path": "annotations/sitcom/groundtruth.txt", "object_class": "sitcom"}, {"name": "skiing", "path": "sequences/skiing", "startFrame": 1, "endFrame": 2654, "nz": 8, "ext": "jpg", "anno_path": "annotations/skiing/groundtruth.txt", "object_class": "skiing"}, {"name": "sup", "path": "sequences/sup", "startFrame": 1, "endFrame": 3506, "nz": 8, "ext": "jpg", "anno_path": "annotations/sup/groundtruth.txt", "object_class": "sup"}, {"name": "tightrope", "path": "sequences/tightrope", "startFrame": 1, "endFrame": 2291, "nz": 8, "ext": "jpg", "anno_path": "annotations/tightrope/groundtruth.txt", "object_class": "tightrope"}, {"name": "uav1", "path": "sequences/uav1", "startFrame": 1, "endFrame": 3469, "nz": 8, "ext": "jpg", "anno_path": "annotations/uav1/groundtruth.txt", "object_class": "uav1"}, {"name": "volkswagen", "path": "sequences/volkswagen", "startFrame": 1, "endFrame": 8576, "nz": 8, "ext": "jpg", "anno_path": "annotations/volkswagen/groundtruth.txt", "object_class": "volkswagen"}, {"name": "warmup", "path": "sequences/warmup", "startFrame": 1, "endFrame": 3961, "nz": 8, "ext": "jpg", "anno_path": "annotations/warmup/groundtruth.txt", "object_class": "warmup"}, {"name": "wingsuit", "path": "sequences/wingsuit", "startFrame": 1, "endFrame": 2508, "nz": 8, "ext": "jpg", "anno_path": "annotations/wingsuit/groundtruth.txt", "object_class": "wingsuit"}, {"name": "yamaha", "path": "sequences/yamaha", "startFrame": 1, "endFrame": 3143, "nz": 8, "ext": "jpg", "anno_path": "annotations/yamaha/groundtruth.txt", "object_class": "yamaha"}]
        return sequence_info_list