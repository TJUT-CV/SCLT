import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_webcam(tracker_name, tracker_param, debug=None, visdom_info=None,gpu_id=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    visdom_info = {} if visdom_info is None else visdom_info
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_webcam(debug, visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--gpu_id', type=str, default='0', help='Specify running GPU.')
    args = parser.parse_args()

    visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}
    run_webcam(args.tracker_name, args.tracker_param, args.debug, visdom_info,args.gpu_id)


if __name__ == '__main__':
    main()