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
from automated_test_util import *
import tensorflow as tf

import oneflow as flow
import oneflow.unittest
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def compare_with_tensorflow(
        device_type, data_type, label_type, batch_size, num_classes, 
    ):
        data_type = type_name_to_flow_type[data_type]
        label_type = type_name_to_flow_type[label_type]
        np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
        np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)

        of_logits = flow.tensor(np_logits, device=device_type, dtype=data_type, requires_grad=True)
        of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
        of_output = flow.nn.functional.sparse_softmax_cross_entropy_with_logits(
            labels=of_labels, logits=of_logits
        ).to(device_type)
        of_output.sum().backward()

        with tf.GradientTape(persistent=True) as tape:
            tf_logits = tf.Variable(np_logits)
            tf_output = tf.nn.sparse_softmax_cross_entropy_with_logits(np_labels, tf_logits)
        tf_logits_diff = tape.gradient(tf_output, tf_logits)

        assert np.allclose(of_output.numpy(), tf_output.numpy(), rtol=1e-03, atol=1e-04)
        assert np.allclose(
            of_logits.grad.numpy(), tf_logits_diff.numpy(), rtol=1e-03, atol=1e-04
        )

class TestSparseSoftmaxCrossEntropyWithLogitsGrid(flow.unittest.TestCase):
    def test_sparse_softmax_cross_entropy_with_logits(test_case):
        np_logits = np.array(
            [
                [2.0, -5.0, 0.5, -0.1],
                [0.0, 0.0, 1.9, 1.4],
                [-100.0, 100.0, -100.0, -100.0],
            ]
        )
        np_labels = np.array([0, 3, 1])
        np_groundtruth = np.array([0.29750752, 1.1448325, 0.0])

        logits = flow.tensor(np_logits, dtype=flow.float32)
        labels = flow.tensor(np_labels, dtype=flow.int32)
        output = flow.nn.functional.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )

        test_case.assertTrue(
            np.allclose(output.numpy(), np_groundtruth, rtol=1e-3, atol=1e-4)
        )

    def test_sparse_softmax_cross_entropy_with_logits(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["label_type"] = ["int32", "int64"]
        arg_dict["batch_size"] = [64, 16]
        arg_dict["num_classes"] = [100, 1000]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
