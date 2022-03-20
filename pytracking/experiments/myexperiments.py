from pytracking.evaluation import Tracker, get_dataset, trackerlist


def sclt_lasot_uav():
    # Run three runs of SCLT on LaSOT and TLP datasets
    trackers = trackerlist('sclt', 'sclt', range(1))
    dataset = get_dataset('lasot', 'tlp')
    return trackers, dataset


def uav_test():
    # Run SCLT on the UAV dataset
    trackers = trackerlist('sclt', 'sclt', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset
