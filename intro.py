import argparse
from typing import List
import os
import sys

import torch
import triton
import triton.language as tl

from display import print_end_line


@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    # print works in the interpreter
    print(range)
    # load the first 5 elements of x_ptr, if the index is less than 5
    # if the index is not less than 5, load 3
    x = tl.load(x_ptr + range, range < 5, 3)
    print(x)


def run_demo1():
    print("Demo1 Output: ")
    demo1[(1, 1, 1)](torch.ones(4, 3))
    print_end_line()


@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    # print works in the interpreter
    print(range)
    mask = (i_range < 4) & (j_range < 3)
    print(mask)
    x = tl.load(x_ptr + range, mask, 2)
    print(x)


def run_demo2():
    print("Demo2 Output: ")
    demo2[(1, 1, 1)](torch.ones(4, 4))
    print_end_line()


@triton.jit
def demo3(z_ptr):
    range = tl.arange(0, 4) * 2
    x = tl.load(z_ptr + range)
    y = x + 10.0
    print(y)
    z = tl.store(z_ptr + range, y, range < 4)


def run_demo3():
    print("Demo3 Output: ")
    z = torch.ones(4, 3)
    demo3[(1, 1, 1)](z)
    print(z)
    print_end_line()


@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20, 2)
    print("Print for each", pid, x)


def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)
    print_end_line()


if __name__ == "__main__":
    run_demo1()
