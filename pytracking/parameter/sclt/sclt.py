from torchvision import transforms
from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 22*16
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    params.net = NetWithBackbone(net_path='super_dimp.pth.tar',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'






    # -------------- new parameters --------------- #
    # parameters for stre and sppp
    params.score_map_input_size=224
    params.stable_len=20
    params.global_search_num=6
    params.global_search_top_num=6
    params.frame_size_thre=500
    params.frame_size_thre_rate=3
    params.frame_size_thre_rate_out_view=2
    params.target_to_frame_bounding_rate=0.005
    params.target_to_frame_in_view_rate=0.01
    params.similarity_super_thre=10
    params.stre_type_classes_num=2
    params.stre_begin_frame_num=20
    params.stre_begin_frame_num2=5
    params.show_step_num=20
    params.state_step_num=10
    params.global_search_step_num=50
    params.global_search_step_num2=80
    params.global_ver_update_tracker_rate = 0.9
    params.search_step_num=20
    params.search_step_num2=10
    params.state_global_step_num=50
    params.state_step_rate=0.8
    params.global_step_nm=100
    params.global_step_rate=0.9
    params.metric_size=107
    params.ver_update_tracker_rate=0.8
    params.verification_lost_rate=0.7
    params.memory_size=200
    params.update_grade1=1.5
    params.update_grade2=2.5
    params.update_grade3=3
    params.sclt_update_grade1=2
    params.sclt_update_grade2=2.5
    params.sclt_update_grade3=3
    params.temp_score_rate=1.5
    params.temp_score_rate2=2
    params.global_step_num=10.8
    params.global_step_num2=5
    params.menory_reduce=0.3
    params.advance_len = 100
    params.advance_rate = 0.8
    params.advance_update_model_rate = 0.9
    params.menory_smallest_len = 5
    params.menory_similar_len = 5
    params.dist_menory_num = 30
    params.model_update_num = 30
    params.model_update_rate = 0.7
    params.lost_global_num = 100
    params.lost_global_rate = 0.8
    params.end_dist_num = 300
    params.votlt_flag = False
    params.stre_model=r'stre_model.pth'
    params.global_track_pth_dir=r'qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
    params.metric_path = r'metric_model.pth'
    params.data_transform = transforms.Compose(
        [transforms.Resize((params.score_map_input_size, params.score_map_input_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return params
