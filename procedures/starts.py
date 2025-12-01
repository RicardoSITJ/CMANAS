##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch, random, PIL, copy, numpy as np
from os import path as osp
from shutil import copyfile


def seed_everything(seed: int):
    # 1. Python & NumPy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # 2. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3. Determinism flags
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 4. Strict Determinism (Optional: comment out if it crashes)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

    print(f"[INFO] Seeding everything with seed {seed}")


def seed_worker(worker_id):
    """
    Ensures that data loader workers have deterministic seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
    args = copy.deepcopy(xargs)
    from log_utils import Logger

    logger = Logger(args.save_dir, args.rand_seed)
    logger.log("Main Function with logger : {:}".format(logger))
    logger.log("Arguments : -------------------------------")
    for name, value in args._get_kwargs():
        logger.log("{:16} : {:}".format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger


def get_machine_info():
    info = "Python  Version  : {:}".format(sys.version.replace("\n", " "))
    info += "\nPillow  Version  : {:}".format(PIL.__version__)
    info += "\nPyTorch Version  : {:}".format(torch.__version__)
    info += "\ncuDNN   Version  : {:}".format(torch.backends.cudnn.version())
    info += "\nCUDA available   : {:}".format(torch.cuda.is_available())
    info += "\nCUDA GPU numbers : {:}".format(torch.cuda.device_count())
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        info += "\nCUDA_VISIBLE_DEVICES={:}".format(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        info += "\nDoes not set CUDA_VISIBLE_DEVICES"
    return info


def save_checkpoint(state, filename, logger):
    if osp.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(
                "Find {:} exist, delete is at first before saving".format(filename)
            )
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(
        filename
    ), "save filename : {:} failed, which is not found.".format(filename)
    if hasattr(logger, "log"):
        logger.log("save checkpoint into {:}".format(filename))
    return filename


def copy_checkpoint(src, dst, logger):
    if osp.isfile(dst):
        if hasattr(logger, "log"):
            logger.log("Find {:} exist, delete is at first before saving".format(dst))
        os.remove(dst)
    copyfile(src, dst)
    if hasattr(logger, "log"):
        logger.log("copy the file from {:} into {:}".format(src, dst))
