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
import oneflow as flow


def sparse_softmax_cross_entropy_with_logits(labels, logits):
    """The interface is consistent with TensorFlow. 
    The documentation is referenced from: 
    https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits

    Computes sparse softmax cross entropy between `logits` and `labels`.

    Measures the probability error in discrete classification tasks in which the
    classes are mutually exclusive (each entry is in exactly one class).  For
    example, each CIFAR-10 image is labeled with one and only one label: an image
    can be a dog or a truck, but not both.

    Note:  For this operation, the probability of a given label is considered
    exclusive.  That is, soft classes are not allowed, and the `labels` vector
    must provide a single specific index for the true class for each row of
    `logits` (each minibatch entry).  For soft softmax classification with
    a probability distribution for each entry, see
    `softmax_cross_entropy_with_logits_v2`.

    Warning: This op expects unscaled logits, since it performs a `softmax`
    on `logits` internally for efficiency.  Do not call this op with the
    output of `softmax`, as it will produce incorrect results.

    A common use case is to have logits of shape
    `[batch_size, num_classes]` and have labels of shape
    `[batch_size]`, but higher dimensions are supported, in which
    case the `dim`-th dimension is assumed to be of size `num_classes`.
    `logits` must have the dtype of `float16`, `float32`, or `float64`, and
    `labels` must have the dtype of `int32` or `int64`.

    >>> logits = tf.constant([[2., -5., .5, -.1],
    ...                       [0., 0., 1.9, 1.4],
    ...                       [-100., 100., -100., -100.]])
    >>> labels = tf.constant([0, 3, 1])
    >>> tf.nn.sparse_softmax_cross_entropy_with_logits(
    ...     labels=labels, logits=logits).numpy()
    array([0.29750752, 1.1448325 , 0.        ], dtype=float32)

    To avoid confusion, passing only named arguments to this function is
    recommended.

    Args:
        labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
        `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
        must be an index in `[0, num_classes)`. Other values will raise an
        exception when this op is run on CPU, and return `NaN` for corresponding
        loss and gradient rows on GPU.
        logits: Unscaled log probabilities of shape `[d_0, d_1, ..., d_{r-1},
        num_classes]` and dtype `float16`, `float32`, or `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of the same shape as `labels` and of the same type as `logits`
        with the softmax cross entropy loss.

    Raises:
        ValueError: If logits are scalars (need to have rank >= 1) or if the rank
        of the labels is not equal to the rank of the logits minus one.

    Examples::

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.arange(1., 7).reshape((1, 2, 3)), dtype=flow.float32)
        >>> output = flow.nn.functional.affine_grid(input, flow.Size([1, 1, 2, 2]), align_corners=True)
        >>> output
        tensor([[[[ 0., -3.],
                  [ 2.,  5.]],
        <BLANKLINE>
                 [[ 4.,  7.],
                  [ 6., 15.]]]], dtype=oneflow.float32)
    """
    (_, out) = flow._C.sparse_softmax_cross_entropy(logits, labels)
    return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
