import torch
import time


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return device


class CustomTimer:

    def __init__(self):
        self.start_perf_count = None
        self.start_process_time = None

    def start(self):
        self.start_perf_count = time.perf_counter()
        self.start_process_time = time.process_time()

    def stop(self, logger, len_dataset):
        elapsed_perf_count = time.perf_counter() - self.start_perf_count
        elapsed_process_time = time.process_time() - self.start_process_time
        logger.info(f"Perf counter: {elapsed_perf_count:.2f} s")
        logger.info(f"  Time per image: {elapsed_perf_count / len_dataset * 1000:.2f} ms")
        logger.info(f"Process time: {elapsed_process_time:.2f} s")
        logger.info(f"  Time per image: {elapsed_process_time / len_dataset * 1000:.2f} ms")


def log_memory(logger):
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f"Current allocated: {allocated:.2f} GB")
    logger.info(f"Current reserved: {reserved:.2f} GB")
    logger.info(f"Peak allocated: {peak:.2f} GB")
