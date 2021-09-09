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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/collective_boxing/runtime_request_info.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

class RequestEntry final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestEntry);
  RequestEntry(const RequestDesc& desc);
  ~RequestEntry() = default;

  const RequestDesc& desc() const { return desc_; }
  int32_t LocalRankCount() const { return local_rank2global_rank_.size(); }
  int32_t LocalRankToGlobalRank(int32_t local_rank) const {
    return local_rank2global_rank_.at(local_rank);
  }
  int32_t GlobalRankToLocalRank(int32_t global_rank) const {
    return global_rank2local_rank_.at(global_rank);
  }
  bool HasRankOnThisNode() const { return !local_rank2global_rank_.empty(); }
  int32_t NodeCount() const { return node_count_; }
  const DeviceDesc& LocalDeviceDesc(int32_t local_rank) const {
    return local_device_vec_.at(local_rank);
  }
  bool IsRootOnThisNode() const {
    return (!local_rank2global_rank_.empty()) && local_rank2global_rank_.front() == 0;
  }

  bool AddRuntimeRequest(int32_t local_rank,
                         std::shared_ptr<const RuntimeRequestInfo> runtime_request_info);
  const std::shared_ptr<const RuntimeRequestInfo>& GetRuntimeRequest(int32_t local_rank);
  std::vector<std::shared_ptr<const RuntimeRequestInfo>> ResetRuntimeRequest();
  int64_t elem_cnt() const { return elem_cnt_; }
  int64_t size_in_bytes() const { return size_in_bytes_; }
  const Symbol<DeviceSet>& device_set_symbol() const { return device_set_symbol_; }

 private:
  RequestDesc desc_;
  int32_t node_count_;
  std::vector<DeviceDesc> local_device_vec_;
  std::vector<int64_t> local_rank2global_rank_;
  std::map<int64_t, int64_t> global_rank2local_rank_;
  int64_t elem_cnt_;
  int64_t size_in_bytes_;
  Symbol<DeviceSet> device_set_symbol_;

  struct State {
    std::vector<std::shared_ptr<const RuntimeRequestInfo>> runtime_request_info_vec;
    int32_t runtime_request_count;
    std::mutex mutex;
  };

  State state_;
};

class RequestStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestStore);
  RequestStore() = default;
  ~RequestStore() = default;

  void AddPlan(const CollectiveBoxingPlan& collective_boxing_plan);
  void DeletePlan(const std::vector<int64_t>& job_ids);

  RequestEntry* MutRequestEntry(int64_t job_id, int32_t request_id) {
    auto it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    return it->second.at(request_id).get();
  }

  void ForEachMutRequestEntryForIdsInJob(
      int64_t job_id, const std::vector<int32_t>& request_ids,
      const std::function<void(RequestEntry*, int32_t i, int32_t request_id)>& Handler) {
    auto it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    for (int32_t i = 0; i < request_ids.size(); ++i) {
      Handler(it->second.at(request_ids.at(i)).get(), i, request_ids.at(i));
    }
  }

  void ForEachMutRequestEntryInJob(
      int64_t job_id,
      const std::function<void(RequestEntry*, int32_t i, int32_t request_id)>& Handler) {
    auto it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    for (int32_t i = 0; i < it->second.size(); ++i) {
      int32_t request_id = i;
      Handler(it->second.at(request_id).get(), i, request_id);
    }
  }

  int32_t RequestCount4Job(int64_t job_id) const {
    const auto& it = job_id2request_entry_vec_.find(job_id);
    CHECK(it != job_id2request_entry_vec_.end());
    return it->second.size();
  }

  int32_t MaxMultiNodeRequestId4Job(int64_t job_id) const {
    const auto& it = job_id2max_multi_node_request_id_.find(job_id);
    CHECK(it != job_id2max_multi_node_request_id_.end());
    return it->second;
  }

  std::pair<int64_t, int32_t> GetJobId7RequestIdByName(const std::string& name) const {
    return name2job_id7request_id_.at(name);
  }

  void* CreateRequestEntryToken(int64_t job_id, int32_t request_id);

  void DestroyRequestToken(void* token);

  RequestEntry* GetRequestEntry(void* token);

 private:
  HashMap<int64_t, std::vector<std::unique_ptr<RequestEntry>>> job_id2request_entry_vec_;
  HashMap<int64_t, int32_t> job_id2max_multi_node_request_id_;
  HashMap<std::string, std::pair<int64_t, int32_t>> name2job_id7request_id_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_REQUEST_STORE_H_
