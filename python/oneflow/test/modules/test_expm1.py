"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_expm1_impl(test_case, device, shape):
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.expm1(x)
    np_out = np.expm1(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.exp(x.numpy()), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestExpm1Module(flow.unittest.TestCase):
    def test_expm1(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_expm1_impl]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest()
    def test_expm1_flow_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = torch.expm1(input)
        return y

    @autotest()
    def test_expm1_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.expm1()
        return y

    @autotest(auto_backward=False)
    def test_expm1_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.expm1(x)
        return y

    @autotest()
    def test_flow_inplace_expm1_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.expm1_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest()
    def test_tensor_inplace_expm1_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.expm1_()
        test_case.assertTrue(id_x == id(x))
        return x


if __name__ == "__main__":
    unittest.main()
