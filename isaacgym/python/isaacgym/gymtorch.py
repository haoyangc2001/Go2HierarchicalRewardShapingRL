"""Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


PyTorch tensor interop """

from . import gymapi

import torch


def _import_gymtorch():
    import os
    import importlib
    import torch.utils.cpp_extension

    global torch
    ver = torch.__version__.split(sep='.')
    ver_major = int(ver[0])
    ver_minor = int(ver[1])

    print("PyTorch version", torch.__version__)

    import torch.cuda
    print("Device count", torch.cuda.device_count())

    thisdir = os.path.dirname(__file__)
    srcdir = os.path.join(thisdir, "_bindings/src/gymtorch")
    print(srcdir)

    sources = [
        os.path.join(srcdir, "gymtorch.cpp")
    ]

    if os.name == "nt":
        cflags = ["/DTORCH_MAJOR=%d" % ver_major, "/DTORCH_MINOR=%d" % ver_minor]
    else:
        cflags = ["-DTORCH_MAJOR=%d" % ver_major, "-DTORCH_MINOR=%d" % ver_minor]

    gt = torch.utils.cpp_extension.load(name="gymtorch", sources=sources, extra_cflags=cflags, verbose=True)
    # print(gt)

    # import all module symbols
    if hasattr(gt, "__all__"):
        attrs = {key: getattr(gt, key) for key in gt.__all__}
    else:
        attrs = {key: value for key, value in gt.__dict__.items() if key[0] != "_"}
    globals().update(attrs)


def _create_context(device='cuda:0'):
    """Force PyTorch to create a primary CUDA context on the specified device"""
    torch.zeros([1], device=device)


def wrap_tensor(gym_tensor, offsets=None, counts=None):
    data = gym_tensor.data_ptr
    device = gym_tensor.device
    dtype = int(gym_tensor.dtype)
    shape = gym_tensor.shape
    if offsets is None:
        offsets = tuple([0] * len(shape))
    if counts is None:
        counts = shape
    return wrap_tensor_impl(data, device, dtype, shape, offsets, counts)


def _torch2gym_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return gymapi.DTYPE_FLOAT32
    elif torch_dtype == torch.uint8:
        return gymapi.DTYPE_UINT8
    elif torch_dtype == torch.int16:
        return gymapi.DTYPE_INT16
    elif torch_dtype == torch.int32:
        return gymapi.DTYPE_UINT32
    elif torch_dtype == torch.int64:
        return gymapi.DTYPE_UINT64
    else:
        raise Exception("Unsupported Gym tensor dtype")


def _torch2gym_device(torch_device):
    if torch_device.type == "cpu":
        return -1
    elif torch_device.type == "cuda":
        return torch_device.index
    else:
        raise Exception("Unsupported Gym tensor device")


def unwrap_tensor(torch_tensor):
    """将PyTorch张量转换为Isaac Gym的Tensor对象，实现零拷贝数据共享。
    参数:
        torch_tensor (torch.Tensor): 要转换的PyTorch张量
    返回:
        gymapi.Tensor: 转换后的Isaac Gym张量对象
    异常:
        Exception: 当输入张量不是连续的(contiguous)时抛出异常
    注意:
        - 此函数创建的Gym张量不拥有底层数据(own_data=False)，数据所有权仍归PyTorch张量所有
        - 确保在使用Gym张量期间保持PyTorch张量不被释放
    """
    # 检查输入张量是否是连续的，Isaac Gym要求传入的张量必须是连续的
    if not torch_tensor.is_contiguous():
        raise Exception("Input tensor must be contiguous")
    
    # 创建新的Isaac Gym张量对象
    gym_tensor = gymapi.Tensor()
    
    # 设置张量的设备信息，将PyTorch设备映射到Gym设备
    gym_tensor.device = _torch2gym_device(torch_tensor.device)
    
    # 设置张量的数据类型，将PyTorch数据类型映射到Gym数据类型
    gym_tensor.dtype = _torch2gym_dtype(torch_tensor.dtype)
    
    # 设置张量的形状信息
    gym_tensor.shape = list(torch_tensor.shape)
    
    # 设置张量的数据地址指针，实现零拷贝共享
    gym_tensor.data_address = torch_tensor.data_ptr()
    
    # 标记Gym张量不拥有底层数据，避免重复释放
    gym_tensor.own_data = False
    
    return gym_tensor


_import_gymtorch()
