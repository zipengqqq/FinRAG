import os
import threading
import time


class SnowflakeIdWorker:
    def __init__(self, datacenter_id=0, worker_id=0, epoch_ms=1704067200000):
        self.datacenter_id_bits = 5
        self.worker_id_bits = 5
        self.sequence_bits = 12
        self.max_datacenter_id = (1 << self.datacenter_id_bits) - 1
        self.max_worker_id = (1 << self.worker_id_bits) - 1
        self.datacenter_id = datacenter_id & self.max_datacenter_id
        self.worker_id = worker_id & self.max_worker_id
        self.sequence = 0
        self.epoch = epoch_ms
        self.last_timestamp = -1
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.worker_id_shift = self.sequence_bits
        self.timestamp_left_shift = self.sequence_bits + self.worker_id_bits + self.datacenter_id_bits
        self.sequence_mask = (1 << self.sequence_bits) - 1
        self.lock = threading.Lock()

    def _time_gen(self):
        return int(time.time() * 1000)

    def _til_next_millis(self, last_ts):
        ts = self._time_gen()
        while ts <= last_ts:
            ts = self._time_gen()
        return ts

    def get_id(self):
        with self.lock:
            ts = self._time_gen()
            if ts < self.last_timestamp:
                ts = self._til_next_millis(self.last_timestamp)
            if ts == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.sequence_mask
                if self.sequence == 0:
                    ts = self._til_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = ts
            return ((ts - self.epoch) << self.timestamp_left_shift) | (self.datacenter_id << self.datacenter_id_shift) | (self.worker_id << self.worker_id_shift) | self.sequence


def _create_worker():
    datacenter_id = int(os.getenv("DATACENTER_ID", "0"))
    worker_id = int(os.getenv("WORKER_ID", "0"))
    epoch_env = os.getenv("SNOWFLAKE_EPOCH_MS")
    epoch_ms = int(epoch_env) if epoch_env else 1704067200000
    return SnowflakeIdWorker(datacenter_id, worker_id, epoch_ms)


id_worker = _create_worker()