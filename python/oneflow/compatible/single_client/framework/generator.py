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
import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow


def create_generator(device=None):
    if device is None:
        device = "auto"
    return oneflow._oneflow_internal.create_generator(device)


def default_generator(device=None):
    if device is None:
        device = "auto"
    return oneflow._oneflow_internal.default_generator(device)


def get_rng_state():
    return default_generator().get_state()


def set_rng_state(state):
    return default_generator().set_state(state)


def manual_seed(seed):
    oneflow._oneflow_internal.manual_seed(seed)
