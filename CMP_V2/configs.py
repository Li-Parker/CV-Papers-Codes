import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="PSP-Net Network", add_help=False)

    # train settings
    parser.add_argument("--dataset", type=str, default="facade")
    parser.add_argument("--model_name", type=str, default="PSPNet")
    parser.add_argument("--pre_model", type=str, default="ViT-B_16.npz")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--root", type=str, default="D:\Desktop\linweiyan\CV-Papers-Codes\CMP_V2\data_v2/",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--setting_size", type=int, default=[1024, 2048],
                        help="original size of data set image.")
    parser.add_argument("--crop_size", type=int, default=[640, 640],
                        help="crop size for training and inference slice.")
    parser.add_argument("--stride_rate", type=float, default=0.5, help="stride ratio.")
    parser.add_argument("--num_epoch", type=int, default=60, help="Number of training steps.")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # optimizer settings
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay.")

    # VIT settings
    parser.add_argument("--encoder", type=str, default="vit_base_patch16", help="name for encoder")
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="dimension for decoder.")
    parser.add_argument("--decoder_depth", type=int, default=2, help="depth for decoder.")
    parser.add_argument("--decoder_num_head", type=int, default=8, help="head number for decoder.")

    # other settings
    parser.add_argument("--save_summary", type=str, default="save_model")
    parser.add_argument("--print_freq", type=str, default=5, help="print frequency.")
    parser.add_argument('--output_dir', default='save_model/', help='path where to save, empty for no saving')
    parser.add_argument("--use_ignore", type=bool, default=False)

    # distributed training parameters
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--device", type=str, default='cuda', help="choose gpu device.")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # distributed training parameters
    parser.add_argument('--eval_img', default='D:\Desktop\linweiyan\CV-Papers-Codes\CMP_V2\data_v2/translated_data/images/IMG_1277.png')

    return parser