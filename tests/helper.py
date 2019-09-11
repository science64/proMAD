import hashlib
import io
import sys
from pathlib import Path


def get_stdout(func, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    old_state = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    func(*args, **kwargs)
    sys.stdout = old_state
    return captured_output.getvalue()


def hash_file(path, skip=16):
    path = Path(path)
    file_hash = hashlib.sha3_256()
    with path.open('rb') as stream:
        # skip ahead to avoid false positives caused by
        # the timestamp in the header
        stream.seek(skip*1024)
        while True:
            data = stream.read(64*1024)
            if not data:
                break
            file_hash.update(data)
    return file_hash.hexdigest()


def hash_mem(mem, skip=16):
    mem.seek(skip * 1024)
    mem_hash = hashlib.sha3_256(mem.read())
    return mem_hash.hexdigest()


def hash_string_mem(mem):
    mem.seek(0)
    mem_hash = hashlib.sha3_256(mem.read().encode('utf-8'))
    return mem_hash.hexdigest()


def hash_array(array):
    array_hash = hashlib.sha3_256(array.tobytes())
    return array_hash.hexdigest()


def hash_bytes(b):
    array_hash = hashlib.sha3_256(b)
    return array_hash.hexdigest()
