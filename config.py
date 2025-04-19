import argparse
from defaults import (
    LANGUAGE_PAIR as DEFAULT_LANGUAGE_PAIR,
    D_MODEL as DEFAULT_D_MODEL,
    LEARNING_RATE as DEFAULT_LEARNING_RATE,
    BETA1 as DEFAULT_BETA1,
    BETA2 as DEFAULT_BETA2,
    EPS as DEFAULT_EPS,
    LR_REDUCTION_FACTOR as DEFAULT_LR_REDUCTION_FACTOR,
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    NUM_WARMUP as DEFAULT_NUM_WARMUP,
    NUM_EPOCHS as DEFAULT_NUM_EPOCHS,
)

DEFAULT_LANGUAGE_SRC, DEFAULT_LANGUAGE_TRG = DEFAULT_LANGUAGE_PAIR


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # shows defaults nicely
    )

    # Data
    data_grp = parser.add_argument_group("Data parameters")
    data_grp.add_argument(
        "--src_lang",
        type=str,
        default=DEFAULT_LANGUAGE_SRC,
        choices=["en", "de"],
        help="Source language code",
    )
    data_grp.add_argument(
        "--trg_lang",
        type=str,
        default=DEFAULT_LANGUAGE_TRG,
        choices=["en", "de"],
        help="Target language code",
    )
    data_grp.add_argument(
        "--d_model", type=int, default=DEFAULT_D_MODEL, help="Model dimension"
    )

    # Optimiser
    opt_grp = parser.add_argument_group("Optimiser parameters")
    opt_grp.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for Adam",
    )
    opt_grp.add_argument(
        "--beta1", type=float, default=DEFAULT_BETA1, help="β1 for Adam"
    )
    opt_grp.add_argument(
        "--beta2", type=float, default=DEFAULT_BETA2, help="β2 for Adam"
    )
    opt_grp.add_argument("--eps", type=float, default=DEFAULT_EPS, help="ε for Adam")

    # Scheduler
    sch_grp = parser.add_argument_group("Scheduler parameters")
    sch_grp.add_argument(
        "--lr_reduction_factor",
        type=float,
        default=DEFAULT_LR_REDUCTION_FACTOR,
        help="Factor for ReduceLROnPlateau",
    )

    # Training / Validation / Test
    train_grp = parser.add_argument_group("Training parameters")
    train_grp.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size"
    )
    train_grp.add_argument(
        "--num_warmup",
        type=int,
        default=DEFAULT_NUM_WARMUP,
        help="Number of warm-up steps",
    )
    train_grp.add_argument(
        "--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Epochs to train"
    )

    val_grp = parser.add_argument_group("Validation parameters")
    val_grp.add_argument(
        "--valid_batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Validation batch size (currently unused)",
    )

    test_grp = parser.add_argument_group("Test parameters")
    test_grp.add_argument(
        "--test_batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Test batch size (currently unused)",
    )

    # Misc
    misc_grp = parser.add_argument_group("Miscellaneous")
    misc_grp.add_argument(
        "--exp_path",
        type=str,
        default=None,
        help="Path to experiment directory (used by test.py / plot.py)",
    )

    return parser.parse_args()


# Parse arguments and export for backward‑compat
args = get_args()
LANGUAGE_PAIR = (args.src_lang, args.trg_lang)
D_MODEL = args.d_model
LEARNING_RATE = args.learning_rate
BETA1, BETA2, EPS = args.beta1, args.beta2, args.eps
LR_REDUCTION_FACTOR = args.lr_reduction_factor
BATCH_SIZE = args.batch_size
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.valid_batch_size
NUM_WARMUP, NUM_EPOCHS = args.num_warmup, args.num_epochs
EXP_PATH = args.exp_path
