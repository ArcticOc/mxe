import argparse

# Constants for default values
DEFAULT_DATA_PATH = "data/mini_imagenet"
DEFAULT_VAL_RESIZE_SIZE = 84
DEFAULT_VAL_CROP_SIZE = 84
DEFAULT_TRAIN_CROP_SIZE = 84
DEFAULT_MODEL = "resnet12"
DEFAULT_PROJECTION = True
DEFAULT_PROJECTION_FEAT_DIM = 128
DEFAULT_MODEL_EMA_STEPS = 32
DEFAULT_MODEL_EMA_DECAY = 0.99998
DEFAULT_AMP = True
DEFAULT_DEVICE = "cuda"
DEFAULT_WORKERS = 16
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 120
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LR = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_LR_SCHEDULER = "steplr"
DEFAULT_LR_STEP_SIZE = 84  # Actually, lr decreases at step_size+warmup_epochs
DEFAULT_LR_GAMMA = 0.1
DEFAULT_LR_MIN = 0.0
DEFAULT_LR_WARMUP_EPOCHS = 10
DEFAULT_LR_WARMUP_METHOD = "linear"
DEFAULT_LR_WARMUP_DECAY = 0.033
DEFAULT_CLIP_GRAD_NORM = None
DEFAULT_BACKEND = "PIL"
DEFAULT_CLASS_AWARE_SAMPLER = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_RESUME = ""
DEFAULT_PRINT_FREQ = 10
DEFAULT_VAL_FREQ = 5
DEFAULT_START_EPOCH = 0
DEFAULT_AUTO_AUGMENT = None
DEFAULT_RA_MAGNITUDE = 9
DEFAULT_AUGMIX_SEVERITY = 3
DEFAULT_RANDOM_ERASE = 0.0
DEFAULT_RA_REPS = 3
DEFAULT_INTERPOLATION = "bilinear"
DEFAULT_WORLD_SIZE = 1
DEFAULT_DIST_URL = "env://"
DEFAULT_LOSS = "PPLoss"
DEFAULT_LOGIT = "l2_dist"
DEFAULT_LOGIT_TEMPERATURE = 1
DEFAULT_NUM_NN = 1
DEFAULT_TEST_ITER = 10000
DEFAULT_VAL_ITER = 3000
DEFAULT_TEST_WAY = 5
DEFAULT_TEST_QUERY = 15
DEFAULT_SHOT = "1,5"
DEFAULT_EVAL_NORM_TYPE = "CCOS"
DEFAULT_CLASS_PROXY = True

# Flags for actions


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="PyTorch Few-Shot Training", add_help=add_help
    )

    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH, type=str, help="dataset path"
    )
    parser.add_argument(
        "--val-resize-size",
        default=DEFAULT_VAL_RESIZE_SIZE,
        type=int,
        help="the resize size used for validation (default: 84)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=DEFAULT_VAL_CROP_SIZE,
        type=int,
        help="the central crop size used for validation (default: 84)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=DEFAULT_TRAIN_CROP_SIZE,
        type=int,
        help="the random crop size used for training (default: 84)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str, help="model name")
    parser.add_argument(
        "--projection",
        default=DEFAULT_PROJECTION,
        type=bool,
        help="Use projection network",
    )
    parser.add_argument(
        "--projection-feat-dim",
        default=DEFAULT_PROJECTION_FEAT_DIM,
        type=int,
        help="Feature dimensionality of output of projection network",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        default=DEFAULT_MODEL_EMA_STEPS,
        type=int,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        default=DEFAULT_MODEL_EMA_DECAY,
        type=float,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--amp",
        default=DEFAULT_AMP,
        type=bool,
        help="Use torch.cuda.amp for mixed precision training",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=DEFAULT_EPOCHS,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--opt", default=DEFAULT_OPTIMIZER, type=str, help="optimizer")
    parser.add_argument(
        "--lr", default=DEFAULT_LR, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=DEFAULT_MOMENTUM, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=DEFAULT_WEIGHT_DECAY,
        type=float,
        metavar="W",
        help="weight decay (default: 0.0005)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler",
        default=DEFAULT_LR_SCHEDULER,
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-step-size",
        default=DEFAULT_LR_STEP_SIZE,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=DEFAULT_LR_GAMMA,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=DEFAULT_LR_MIN,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=DEFAULT_LR_WARMUP_EPOCHS,
        type=int,
        help="the number of epochs to warmup (default: 10)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default=DEFAULT_LR_WARMUP_METHOD,
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument(
        "--lr-warmup-decay",
        default=DEFAULT_LR_WARMUP_DECAY,
        type=float,
        help="the decay for lr",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=DEFAULT_CLIP_GRAD_NORM,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        type=str.lower,
        help="PIL or tensor - case insensitive",
    )
    parser.add_argument(
        "--class-aware-sampler",
        type=str,
        default=DEFAULT_CLASS_AWARE_SAMPLER,
        help="uniform class-aware sampler",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=str,
        help="path to save outputs",
    )
    parser.add_argument(
        "--resume", default=DEFAULT_RESUME, type=str, help="path of checkpoint"
    )
    parser.add_argument(
        "--print-freq", default=DEFAULT_PRINT_FREQ, type=int, help="print frequency"
    )
    parser.add_argument(
        "--val-freq", default=DEFAULT_VAL_FREQ, type=int, help="validation frequency"
    )
    parser.add_argument(
        "--start-epoch",
        default=DEFAULT_START_EPOCH,
        type=int,
        metavar="N",
        help="start epoch",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the model",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="enable saving all the checkpoints throughout training",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--auto-augment",
        default=DEFAULT_AUTO_AUGMENT,
        type=str,
        help="auto augment policy (default: None)",
    )
    parser.add_argument(
        "--ra-magnitude",
        default=DEFAULT_RA_MAGNITUDE,
        type=int,
        help="magnitude of auto augment policy",
    )
    parser.add_argument(
        "--augmix-severity",
        default=DEFAULT_AUGMIX_SEVERITY,
        type=int,
        help="severity of augmix policy",
    )
    parser.add_argument(
        "--random-erase",
        default=DEFAULT_RANDOM_ERASE,
        type=float,
        help="random erasing probability (default: 0.0)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=DEFAULT_RA_REPS,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--interpolation",
        default=DEFAULT_INTERPOLATION,
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    parser.add_argument(
        "--world-size",
        default=DEFAULT_WORLD_SIZE,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist-url",
        default=DEFAULT_DIST_URL,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--loss",
        default=DEFAULT_LOSS,
        type=str,
        help="loss function (default: MultiXELoss)",
    )
    parser.add_argument(
        "--logit", default=DEFAULT_LOGIT, type=str, help="logit type (default: l1_dist)"
    )
    parser.add_argument(
        "--logit-temperature",
        default=DEFAULT_LOGIT_TEMPERATURE,
        type=float,
        dest="T",
        help="temperature for logit",
    )
    parser.add_argument(
        "--class-proxy",
        default=DEFAULT_CLASS_PROXY,
        type=bool,
        help="Augment support set by trainable class centers",
    )
    parser.add_argument(
        "--num-NN",
        default=DEFAULT_NUM_NN,
        type=int,
        help="number of nearest neighbors, set this number >1 when do kNN (default 1)",
    )
    parser.add_argument(
        "--soft-assignment",
        action="store_true",
        help="use soft assignment for multiple shot classification.",
    )
    parser.add_argument(
        "--median-prototype",
        action="store_true",
        help="use median instead of mean for computing prototypes",
    )
    parser.add_argument(
        "--test-iter",
        default=DEFAULT_TEST_ITER,
        type=int,
        help="number of iterations on test set (default 10000)",
    )
    parser.add_argument(
        "--val-iter",
        default=DEFAULT_VAL_ITER,
        type=int,
        help="number of iterations on val set (default 3000)",
    )
    parser.add_argument(
        "--test-way",
        default=DEFAULT_TEST_WAY,
        type=int,
        help="number of ways during val/test (default 5)",
    )
    parser.add_argument(
        "--test-query",
        default=DEFAULT_TEST_QUERY,
        type=int,
        help="number of queries during val/test (default 15)",
    )
    parser.add_argument(
        "--shot",
        default=DEFAULT_SHOT,
        type=str,
        help="number of shots on test set (default 1,5)",
    )
    parser.add_argument(
        "--eval-norm-type",
        default=DEFAULT_EVAL_NORM_TYPE,
        type=str,
        help="norm type in evaluation [CL2N|L2N|COS|CCOS] (default CL2N)",
    )
    parser.add_argument(
        "--disable-nearest-mean-classifier",
        action="store_true",
        help="not use nearest-mean classifier",
    )

    return parser


args = get_args_parser().parse_args()
