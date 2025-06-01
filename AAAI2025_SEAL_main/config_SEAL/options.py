import argparse

# basic settings
parser = argparse.ArgumentParser(description='SEAL for WTAL')
parser.add_argument('--exp_dir', type=str, default='./outputs', help='the directory of experiments')
parser.add_argument('--pre_data', type=str, default='./outputs', help='the directory of experiments')
parser.add_argument('--run_type', type=str, default='train', help='train or test (default: train)')

# dataset patameters
parser.add_argument('--dataset_name', type=str, default='Thumos14reduced', help='dataset to train on')
parser.add_argument('--dataset_root', type=str, default='./data/Thumos14reduced', help='dataset root path')
parser.add_argument('--descriptors_root', type=str, default='./data/descriptors', help='descriptors root path')
parser.add_argument('--base_method', type=str, default='base', help='baseline S-MIL method name')

# model parameters
parser.add_argument('--model_name', type=str, default='SEAL', help='model name (default: SEAL)')
parser.add_argument('--num_class', type=int, default=20, help='number of classes (default: 20)')
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--roi_size', type=int, default=12, help='roi size for proposal features extraction (default: 12)')
parser.add_argument('--max_proposal', type=int, default=3000, help='maximum number of proposal during training (default: 1000)')
parser.add_argument('--pretrained_ckpt', type=str, default='./outputs/ckpt/best_model_SEAL.pkl', help='ckpt for pretrained model')

# training paramaters
parser.add_argument('--batch_size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight deacy (default: 0.001)')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout value (default: 0.5)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_epoch', type=int, default=200, help='maximum epoch to train (default: 200)')
parser.add_argument('--general_appearance_path', type=str, default='./general_appearance_descriptors', help='Path to general appearance descriptors')
parser.add_argument('--general_motion_path', type=str, default='./general_motion_descriptors', help='Path to general motion descriptors')

# training paramaters for new loss
parser.add_argument('--alpha_1', type=float, default=3.4, help='pc loss (default: 1.5)')
parser.add_argument('--alpha_2', type=int, default=70, help='SC loss (default: 190)')
parser.add_argument('--alpha_3', type=int, default=160, help='SME loss (default: 30)')
parser.add_argument('--alpha_4', type=int, default=8, help='K (default: 3)')
parser.add_argument('--alpha_5', type=float, default=0.6, help='PC2 (default: 0.3)')
parser.add_argument('--alpha_6', type=int, default=9.0, help='PC3 (default: 5)')
    
parser.add_argument('--interval', type=int, default=10, help='epoch interval for performing the test (default: 10)')
parser.add_argument('--k', type=float, default=8, help='top-k for aggregating video-level classification (default: 8)')
parser.add_argument('--gamma', type=float, default=0.8, help='threshold for select pseudo instances (default: 0.8)')

# testing parameters
parser.add_argument('--threshold_cls', type=float, default=0.25, help='video-level classification threshold')

# Pre_features and scores
parser.add_argument('--tcas_output_dir', type=str, default='/mnt/limaodong/outputs/pre_extracted_features_and_scores', help='Directory to save pre-extracted features and scores')


