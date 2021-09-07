/*
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
*/
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COORDINATOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COORDINATOR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class CollectiveBoxingPlan;

namespace boxing {

namespace collective {

class RequestStore;
class Executor;

class Coordinator {
 public:
  Coordinator() = default;
  virtual ~Coordinator() = default;

  virtual void Init(std::shared_ptr<RequestStore> request_store,
                    std::shared_ptr<Executor> executor) = 0;
  virtual void AddPlan(const std::vector<int64_t>& job_ids) = 0;
  virtual void AddRequest(int64_t job_id, int32_t request_id) = 0;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COORDINATOR_H_
