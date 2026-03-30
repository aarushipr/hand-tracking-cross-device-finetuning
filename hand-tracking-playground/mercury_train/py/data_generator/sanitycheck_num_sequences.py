import os
import psutil
import time
import pandas
import subprocess
from io import StringIO
import pandas as pd

sequence_length: float = 200.0


class State:
    superroot: str = "/4/generation_run_jan9"

    last_num_images: int = 0

    # Number of folders in superroot, basically. Either completed or
    # in-progress.
    last_num_sequences: int = 0

    # Think of this as the book-end "before" the stack of uncompleted
    # sequence. So 0 comes before seq0 (seq0 is in progress), 1 comes before
    # seq1, etc.
    last_end_of_completed_stack: int = 0

    last_sample_time: float

    df: pd.DataFrame


def get_num_sequences(st: State):
    print([dir for dir in os.listdir(st.superroot)
          if os.path.isdir(os.path.join(st.superroot, dir))])
    return len([dir for dir in os.listdir(st.superroot)
               if os.path.isdir(os.path.join(st.superroot, dir))])


def get_num_images(st: State):
    s = 0
    num_dirs = get_num_sequences(st)

    s += st.last_end_of_completed_stack * sequence_length

    last_is_completed = True
    for dir_num in range(st.last_end_of_completed_stack, num_dirs):
        try:
            dir = f"seq{dir_num}"
            imgs_folder = os.path.join(st.superroot, dir, "imgs_color")
            num = len(os.listdir(imgs_folder))
            s += num

            if last_is_completed:
                st.last_end_of_completed_stack = dir_num
                this_completed = num == int(sequence_length)
                if not this_completed:
                    # We found the bookend
                    last_is_completed = False
            # print(f"ok, {dir_num}, {st.last_end_of_completed_stack} ")
            print(num)
            # s += len(os.listdir(imgs_folder))
        except FileNotFoundError:
            # we hit a sad place with no directory
            print(f"Didn't find directory on sequence {dir_num}")
            pass
    print(s)
    return s


def get_sequences_numbers(st: State, d: dict):
    for i in range(7):
        d[str(i)] = 0
    for dir_num in range(get_num_sequences(st)):
        dir = f"seq{dir_num}"
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        try:
            with open(os.path.join(st.superroot, dir, "model_idx")) as f:
                l = f.readlines()[0]
                print("l", l)
                s = int(l)
                print("s", s)
                d[str(s)] += 1
        except FileNotFoundError:
            pass


def main():
    st = State()

    st.df = pd.DataFrame(columns=["images_per_second",
                                  "new_sequences",
                                  "gpu utilization %",
                                  "gpu mem used %",
                                  "gpu mem utilization %",
                                  "cpu usage %",
                                  "memory usage %"] + [str(d) for d in range(7)])
    _ = psutil.cpu_percent(interval=1)
    st.last_num_sequences = get_num_sequences(st)
    st.last_num_images = get_num_images(st)

    st.last_sample_time = time.time()

    column = {}

    get_sequences_numbers(st, column)
    print(column)
    # nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory
    # --format=csv,nounits


if __name__ == "__main__":
    print("AAAAAAAAAAAAAAAAAAAA LOADQAWFASDL;FHASDLK;FHASASFDLKJHASLKFHASJKLFHASF")
    main()
