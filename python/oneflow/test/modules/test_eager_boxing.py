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
import oneflow as flow
import os

import oneflow.unittest
from test_util import GenArgList


def _test_eager_boxing_with_non_overlapping_placement_p_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[6, 16], [9, 17], [7, 13], [12, 16],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[15, 27], [19, 5], [11, 9], [15, 4],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_b_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[4, 6], [6, 8], [3, 7], [6, 8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[5, 20], [9, 0], [5, 0], [9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s0_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [2, 10], [3, 9], [4, 6], [6, 8],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [2, 10], [3, 9], [4, 6], [6, 8],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.broadcast)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20],
                        [6, 8, 9, 0],
                        [3, 7, 5, 0],
                        [6, 8, 9, 0],
                        [2, 10, 10, 7],
                        [3, 9, 10, 5],
                        [4, 6, 6, 9],
                        [6, 8, 6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20],
                        [6, 8, 9, 0],
                        [3, 7, 5, 0],
                        [6, 8, 9, 0],
                        [2, 10, 10, 7],
                        [3, 9, 10, 5],
                        [4, 6, 6, 9],
                        [6, 8, 6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_non_overlapping_placement_s1_to_p(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.partial_sum)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20],
                        [6, 8, 9, 0],
                        [3, 7, 5, 0],
                        [6, 8, 9, 0],
                        [2, 10, 10, 7],
                        [3, 9, 10, 5],
                        [4, 6, 6, 9],
                        [6, 8, 6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_overlapping_placement_p_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[15, 20], [16, 19], [13, 16], [15, 23],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[20, 35], [28, 10], [20, 11], [20, 12],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_overlapping_placement_b_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[4, 6], [6, 8], [3, 7], [6, 8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[5, 20], [9, 0], [5, 0], [9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_overlapping_placement_s0_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [4, 6],
                        [6, 8],
                        [3, 7],
                        [6, 8],
                        [2, 10],
                        [3, 9],
                        [4, 6],
                        [6, 8],
                        [9, 4],
                        [7, 2],
                        [6, 3],
                        [3, 7],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                        [5, 8],
                        [9, 5],
                        [9, 2],
                        [5, 8],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_overlapping_placement_s1_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5],
                        [6, 8, 9],
                        [3, 7, 5],
                        [6, 8, 9],
                        [2, 10, 10],
                        [3, 9, 10],
                        [4, 6, 6],
                        [6, 8, 6],
                        [9, 4, 5],
                        [7, 2, 9],
                        [6, 3, 9],
                        [3, 7, 5],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [20, 8, 9],
                        [0, 4, 6],
                        [0, 3, 5],
                        [0, 8, 7],
                        [7, 10, 3],
                        [5, 5, 6],
                        [9, 8, 6],
                        [4, 5, 3],
                        [8, 9, 6],
                        [5, 4, 1],
                        [2, 5, 2],
                        [8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_overlapping_placement_s1_to_s0(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_overlapping_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.broadcast)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_overlapping_placement_s1_to_p(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.partial_sum)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_p_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    new_placement = flow.placement(out_device, {0: [1, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[15, 20], [16, 19], [13, 16], [15, 23],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[20, 35], [28, 10], [20, 11], [20, 12],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_b_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    new_placement = flow.placement(out_device, {0: [1, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[4, 6], [6, 8], [3, 7], [6, 8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[5, 20], [9, 0], [5, 0], [9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_s0_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    new_placement = flow.placement(out_device, {0: [1, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [4, 6],
                        [6, 8],
                        [3, 7],
                        [6, 8],
                        [2, 10],
                        [3, 9],
                        [4, 6],
                        [6, 8],
                        [9, 4],
                        [7, 2],
                        [6, 3],
                        [3, 7],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                        [5, 8],
                        [9, 5],
                        [9, 2],
                        [5, 8],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 2, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [1, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array([[4, 6], [6, 8], [3, 7], [6, 8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array([[5, 20], [9, 0], [5, 0], [9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s0(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 2, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [1, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array([[4, 6, 5, 20], [6, 8, 9, 0],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array([[3, 7, 5, 0], [6, 8, 9, 0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_p(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 2, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [1, 3]})
    z = y.to_consistent(new_placement, flow.sbp.partial_sum)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 2, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [1, 3]})
    z = y.to_consistent(new_placement, flow.sbp.broadcast)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_p_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[15], [16], [13], [15],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[20], [19], [16], [23],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[20], [28], [20], [20],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[35], [10], [11], [12],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_b_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[4], [6], [3], [6],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[6], [8], [7], [8],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[5], [9], [5], [9],], dtype=np.float32,),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array([[20], [0], [0], [0],], dtype=np.float32,),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_s0_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [[4, 6, 5, 20], [6, 8, 9, 0], [3, 7, 5, 0], [6, 8, 9, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [[2, 10, 10, 7], [3, 9, 10, 5], [4, 6, 6, 9], [6, 8, 6, 4]],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [[9, 6, 5, 8], [4, 9, 7, 0], [2, 5, 7, 9], [6, 8, 10, 0]], dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [[9, 4, 5, 8], [7, 2, 9, 5], [6, 3, 9, 2], [3, 7, 5, 8]], dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    y = x.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[4], [6], [3], [6], [2], [3], [4], [6], [9], [7], [6], [3],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[6], [8], [7], [8], [10], [9], [6], [8], [4], [2], [3], [7],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[5], [9], [5], [9], [10], [10], [6], [6], [5], [9], [9], [5],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[20], [0], [0], [0], [7], [5], [9], [4], [8], [5], [2], [8],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_b(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.broadcast)
    test_case.assertEqual(z.placement, new_placement)
    test_case.assertTrue(
        np.array_equal(
            z.to_local().numpy(),
            np.array(
                [
                    [4, 6, 5, 20, 8, 9],
                    [6, 8, 9, 0, 4, 6],
                    [3, 7, 5, 0, 3, 5],
                    [6, 8, 9, 0, 8, 7],
                    [2, 10, 10, 7, 10, 3],
                    [3, 9, 10, 5, 5, 6],
                    [4, 6, 6, 9, 8, 6],
                    [6, 8, 6, 4, 5, 3],
                    [9, 4, 5, 8, 9, 6],
                    [7, 2, 9, 5, 4, 1],
                    [6, 3, 9, 2, 5, 2],
                    [3, 7, 5, 8, 9, 3],
                ],
                dtype=np.float32,
            ),
        )
    )


def _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_p(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.partial_sum)
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    else:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s0(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(0))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 5, 20, 8, 9], [6, 8, 9, 0, 4, 6], [3, 7, 5, 0, 3, 5],],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[6, 8, 9, 0, 8, 7], [2, 10, 10, 7, 10, 3], [3, 9, 10, 5, 5, 6],],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6, 6, 9, 8, 6], [6, 8, 6, 4, 5, 3], [9, 4, 5, 8, 9, 6],],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[7, 2, 9, 5, 4, 1], [6, 3, 9, 2, 5, 2], [3, 7, 5, 8, 9, 3],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s1(
    test_case, in_device, out_device
):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9, 5, 20],
                [6, 8, 9, 0, 4, 6, 9, 0],
                [3, 7, 5, 0, 3, 5, 0, 3],
                [6, 8, 9, 0, 8, 7, 8, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3, 10, 7],
                [3, 9, 10, 5, 5, 6, 9, 10],
                [4, 6, 6, 9, 8, 6, 6, 9],
                [6, 8, 6, 4, 5, 3, 8, 6],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6, 8, 3],
                [4, 9, 7, 0, 2, 1, 9, 7],
                [2, 5, 7, 9, 4, 8, 5, 7],
                [6, 8, 10, 0, 4, 9, 8, 10],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6, 5, 8],
                [7, 2, 9, 5, 4, 1, 7, 2],
                [6, 3, 9, 2, 5, 2, 9, 2],
                [3, 7, 5, 8, 9, 3, 7, 5],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    new_placement = flow.placement(out_device, {0: [0, 1, 2, 3]})
    z = y.to_consistent(new_placement, flow.sbp.split(1))
    test_case.assertEqual(z.placement, new_placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [2, 10], [3, 9], [4, 6], [6, 8],],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 2:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [[8, 9], [4, 6], [3, 5], [8, 7], [10, 3], [5, 6], [8, 6], [5, 3],],
                    dtype=np.float32,
                ),
            )
        )
    elif flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [0, 3],
                        [8, 9],
                        [10, 7],
                        [9, 10],
                        [6, 9],
                        [8, 6],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_p_to_s1(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
                [6, 8, 9, 0, 4, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
                [4, 9, 7, 0, 2, 1],
                [6, 3, 9, 2, 5, 2],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
                [6, 3, 9, 2, 5, 2],
                [2, 5, 7, 9, 4, 8],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
                [7, 2, 9, 5, 4, 1],
                [4, 9, 7, 0, 2, 1],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.partial_sum)
    y = x.to_consistent(placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[15, 20], [16, 19], [13, 16], [15, 23], [17, 19], [16, 20],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[20, 35], [28, 10], [20, 11], [20, 12], [25, 5], [22, 6],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[27, 18], [13, 13], [16, 13], [22, 13], [10, 8], [12, 6],],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_b_to_s1(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
                [6, 8, 9, 0, 4, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
                [4, 9, 7, 0, 2, 1],
                [6, 3, 9, 2, 5, 2],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
                [6, 3, 9, 2, 5, 2],
                [2, 5, 7, 9, 4, 8],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
                [7, 2, 9, 5, 4, 1],
                [4, 9, 7, 0, 2, 1],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.broadcast)
    y = x.to_consistent(placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[4, 6], [6, 8], [3, 7], [6, 8], [6, 8], [6, 8],], dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[5, 20], [9, 0], [5, 0], [9, 0], [9, 0], [6, 4],],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [[8, 9], [4, 6], [3, 5], [8, 7], [4, 6], [5, 3],], dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_s0_to_s1(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    test_case.assertEqual(y.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [4, 6],
                        [6, 8],
                        [3, 7],
                        [6, 8],
                        [2, 10],
                        [3, 9],
                        [4, 6],
                        [6, 8],
                        [9, 4],
                        [7, 2],
                        [6, 3],
                        [3, 7],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                        [5, 8],
                        [9, 5],
                        [9, 2],
                        [5, 8],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                y.to_local().numpy(),
                np.array(
                    [
                        [8, 9],
                        [4, 6],
                        [3, 5],
                        [8, 7],
                        [10, 3],
                        [5, 6],
                        [8, 6],
                        [5, 3],
                        [9, 6],
                        [4, 1],
                        [5, 2],
                        [9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_s1_to_s1(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    z = y.to_consistent(placement, flow.sbp.split(1))
    test_case.assertEqual(z.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6],
                        [6, 8],
                        [3, 7],
                        [6, 8],
                        [2, 10],
                        [3, 9],
                        [4, 6],
                        [6, 8],
                        [9, 4],
                        [7, 2],
                        [6, 3],
                        [3, 7],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [5, 20],
                        [9, 0],
                        [5, 0],
                        [9, 0],
                        [10, 7],
                        [10, 5],
                        [6, 9],
                        [6, 4],
                        [5, 8],
                        [9, 5],
                        [9, 2],
                        [5, 8],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [8, 9],
                        [4, 6],
                        [3, 5],
                        [8, 7],
                        [10, 3],
                        [5, 6],
                        [8, 6],
                        [5, 3],
                        [9, 6],
                        [4, 1],
                        [5, 2],
                        [9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_s1_to_s0(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    z = y.to_consistent(placement, flow.sbp.split(0))
    test_case.assertEqual(z.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_s1_to_p(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    z = y.to_consistent(placement, flow.sbp.partial_sum)
    test_case.assertEqual(z.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_with_same_placement_s1_to_b(test_case, in_device, out_device):
    if flow.env.get_rank() == 0:
        np_arr = np.array(
            [
                [4, 6, 5, 20, 8, 9],
                [6, 8, 9, 0, 4, 6],
                [3, 7, 5, 0, 3, 5],
                [6, 8, 9, 0, 8, 7],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 1:
        np_arr = np.array(
            [
                [2, 10, 10, 7, 10, 3],
                [3, 9, 10, 5, 5, 6],
                [4, 6, 6, 9, 8, 6],
                [6, 8, 6, 4, 5, 3],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 2:
        np_arr = np.array(
            [
                [9, 6, 5, 8, 3, 6],
                [4, 9, 7, 0, 2, 1],
                [2, 5, 7, 9, 4, 8],
                [6, 8, 10, 0, 4, 9],
            ],
            dtype=np.float32,
        )
    elif flow.env.get_rank() == 3:
        np_arr = np.array(
            [
                [9, 4, 5, 8, 9, 6],
                [7, 2, 9, 5, 4, 1],
                [6, 3, 9, 2, 5, 2],
                [3, 7, 5, 8, 9, 3],
            ],
            dtype=np.float32,
        )
    device = flow.device(in_device)
    tensor = flow.tensor(np_arr, device=device, dtype=flow.float32)
    placement = flow.placement(in_device, {0: [0, 1, 3]})
    x = tensor.to_consistent(placement, flow.sbp.split(0))
    y = x.to_consistent(placement, flow.sbp.split(1))
    z = y.to_consistent(placement, flow.sbp.broadcast)
    test_case.assertEqual(z.placement, placement)
    if flow.env.get_rank() == 0:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 1:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    if flow.env.get_rank() == 3:
        test_case.assertTrue(
            np.array_equal(
                z.to_local().numpy(),
                np.array(
                    [
                        [4, 6, 5, 20, 8, 9],
                        [6, 8, 9, 0, 4, 6],
                        [3, 7, 5, 0, 3, 5],
                        [6, 8, 9, 0, 8, 7],
                        [2, 10, 10, 7, 10, 3],
                        [3, 9, 10, 5, 5, 6],
                        [4, 6, 6, 9, 8, 6],
                        [6, 8, 6, 4, 5, 3],
                        [9, 4, 5, 8, 9, 6],
                        [7, 2, 9, 5, 4, 1],
                        [6, 3, 9, 2, 5, 2],
                        [3, 7, 5, 8, 9, 3],
                    ],
                    dtype=np.float32,
                ),
            )
        )


def _test_eager_boxing_b_to_s(
    test_case, shape, device_type, in_device_list, out_device_list
):
    np_arr = np.random.uniform(-1e-05, 1e-05, shape)
    # use cuda to avoid slice boxing here
    placement_with_all_cuda_device = flow.env.all_device_placement("cuda")

    x = flow.tensor(np_arr, device="cuda", dtype=flow.float32)

    x = x.to_consistent(placement_with_all_cuda_device, flow.sbp.broadcast)

    placement = flow.placement(device_type, {0: in_device_list})
    y = x.to_consistent(placement, flow.sbp.broadcast)

    new_placement = flow.placement(device_type, {0: out_device_list})
    z = y.to_consistent(new_placement, flow.sbp.split(0))

    if flow.env.get_rank() in out_device_list:
        idx = out_device_list.index(flow.env.get_rank())
        step = int(shape[0] / len(out_device_list))
        test_case.assertTrue(
            np.allclose(
                z.to_local().numpy(),
                x.to_local().numpy()[idx * step : (idx + 1) * step],
                1e-5,
                1e-5,
            )
        )
    test_case.assertEqual(z.placement, new_placement)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithNonOverlappingPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_non_overlapping_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_p_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_b_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s0_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s1(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_s0(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_b(test_case, *arg)

    def test_eager_boxing_with_non_overlapping_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_non_overlapping_placement_s1_to_p(test_case, *arg)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithOverlappingPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_overlapping_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_p_to_s1(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_b_to_s1(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_s0_to_s1(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_s1_to_s1(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_s1_to_s0(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_s1_to_b(test_case, *arg)

    def test_eager_boxing_with_overlapping_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_overlapping_placement_s1_to_p(test_case, *arg)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithInPlacementContainOutPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_in_placement_contain_out_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_p_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_b_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_s0_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_s0(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_p(
                test_case, *arg
            )

    def test_eager_boxing_with_in_placement_contain_out_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_in_placement_contain_out_placement_s1_to_b(
                test_case, *arg
            )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithOutPlacementContainInPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_out_placement_contain_in_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_p_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_b_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_s0_to_s1(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_b(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_p(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s0(
                test_case, *arg
            )

    def test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_out_placement_contain_in_placement_s1_to_s1(
                test_case, *arg
            )


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingWithSameInOutPlacement(flow.unittest.TestCase):
    def test_eager_boxing_with_same_placement_s0_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_s0_to_s1(test_case, *arg)

    def test_eager_boxing_with_same_placement_p_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_p_to_s1(test_case, *arg)

    def test_eager_boxing_with_same_placement_b_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_b_to_s1(test_case, *arg)

    def test_eager_boxing_with_same_placement_s1_to_s1(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_s1_to_s1(test_case, *arg)

    def test_eager_boxing_with_same_placement_s1_to_s0(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_s1_to_s0(test_case, *arg)

    def test_eager_boxing_with_same_placement_s1_to_p(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_s1_to_p(test_case, *arg)

    def test_eager_boxing_with_same_placement_s1_to_b(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_with_same_placement_s1_to_b(test_case, *arg)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingBToS(flow.unittest.TestCase):
    def test_eager_boxing_b_to_s(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(12, 12), (12, 18, 24)]
        arg_dict["device_type"] = ["cpu", "cuda"]
        arg_dict["in_device_list"] = [[0, 1], [1, 2, 3]]
        arg_dict["out_device_list"] = [[2, 3], [0, 1, 3]]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_b_to_s(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
