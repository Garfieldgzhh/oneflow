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
#include "oneflow/core/job/collective_boxing/scheduler.h"
#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/job/collective_boxing/coordinator.h"
#include "oneflow/core/job/collective_boxing/static_group_coordinator.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/collective_boxing/nccl_executor_backend.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/resource_desc.h"
#ifdef WITH_MPI
#include "oneflow/core/job/collective_boxing/dynamic_coordinator.h"
#endif

namespace oneflow {

namespace boxing {

namespace collective {

class RequestHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RequestHandle);
  RequestHandle(int64_t job_id, int32_t request_id, int32_t local_rank, void* request_entry_token,
                void* request_token, void* executor_token)
      : job_id_(job_id),
        request_id_(request_id),
        local_rank_(local_rank),
        request_entry_token_(request_entry_token),
        request_token_(request_token),
        executor_token_(executor_token) {}
  ~RequestHandle() = default;

  int64_t job_id() const { return job_id_; }

  int32_t request_id() const { return request_id_; }

  int32_t local_rank() const { return local_rank_; }

  void* request_entry_token() { return request_entry_token_; }
  void* request_token() { return request_token_; }
  void* executor_token() { return executor_token_; }

 private:
  int64_t job_id_;
  int32_t request_id_;
  int32_t local_rank_;
  void* request_entry_token_;
  void* request_token_;
  void* executor_token_;
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl() = default;
  ~ExecutorImpl() override = default;

  void Init(std::shared_ptr<RequestStore> request_store) override;
  void AddPlan(const std::vector<int64_t>& job_ids) override;
  void DeletePlan(const std::vector<int64_t>& job_ids) override;
  void GroupRequests(const int64_t job_id, const std::vector<int32_t>& request_ids,
                     const std::function<void(int64_t, std::vector<int32_t>&&)>& Handler) override;
  void ExecuteGroupedRequests(const int64_t job_id, const std::vector<int32_t>& request_ids,
                              void* executor_token) override;
  void* CreateExecutorToken(int64_t job_id, int32_t request_id) override;

 private:
  Backend GetUniqueBackend(int64_t job_id, const std::vector<int32_t>& request_ids);

  std::vector<std::unique_ptr<ExecutorBackend>> backends_;
  std::shared_ptr<RequestStore> request_store_;
  std::vector<int32_t> group_buffer_;
  int64_t group_buffer_job_id_;
};

void ExecutorImpl::Init(std::shared_ptr<RequestStore> request_store) {
  request_store_ = request_store;
  backends_.resize(Backend_ARRAYSIZE);
#ifdef WITH_CUDA
  std::unique_ptr<ExecutorBackend> nccl_backend = std::make_unique<NcclExecutorBackend>();
  nccl_backend->Init(request_store_);
  backends_.at(Backend::kBackendNCCL) = std::move(nccl_backend);
#endif
}

void ExecutorImpl::AddPlan(const std::vector<int64_t>& job_ids) {
  backends_.at(Backend::kBackendNCCL)->AddPlan(job_ids);
}

void ExecutorImpl::DeletePlan(const std::vector<int64_t>& job_ids) {
  backends_.at(Backend::kBackendNCCL)->DeletePlan(job_ids);
}

void* ExecutorImpl::CreateExecutorToken(int64_t job_id, int32_t request_id) {
  return backends_.at(Backend::kBackendNCCL)->CreateExecutorToken(job_id, request_id);
}

void ExecutorImpl::GroupRequests(
    const int64_t job_id, const std::vector<int32_t>& request_ids,
    const std::function<void(int64_t, std::vector<int32_t>&&)>& Handler) {
  if (request_ids.empty()) { return; }
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  if (!conf.enable_fusion()) {
    for (int32_t request_id : request_ids) { Handler(job_id, std::vector<int32_t>({request_id})); }
    return;
  }
  auto HandleGroup = [&]() {
    if (group_buffer_.empty()) { return; }
    if (group_buffer_.size() == 1) {
      Handler(group_buffer_job_id_, std::vector<int32_t>({group_buffer_.front()}));
    } else {
      const auto backend =
          request_store_->MutRequestEntry(group_buffer_job_id_, group_buffer_.front())
              ->desc()
              .op_desc()
              .backend();
      backends_.at(backend)->GroupRequests(group_buffer_job_id_, group_buffer_, Handler);
    }
    group_buffer_.clear();
  };
  for (const int32_t request_id : request_ids) {
    if (!group_buffer_.empty()) {
      const auto* cur_entry = request_store_->MutRequestEntry(job_id, request_id);
      const auto* group_entry =
          request_store_->MutRequestEntry(group_buffer_job_id_, group_buffer_.front());
      if (job_id != group_buffer_job_id_
          || cur_entry->desc().dependency_depth() != group_entry->desc().dependency_depth()
          || cur_entry->desc().op_desc().backend() != group_entry->desc().op_desc().backend()
          || cur_entry->device_set_symbol() != group_entry->device_set_symbol()) {
        HandleGroup();
      }
    }
    group_buffer_.push_back(request_id);
    group_buffer_job_id_ = job_id;
  }
  HandleGroup();
}

void ExecutorImpl::ExecuteGroupedRequests(const int64_t job_id,
                                          const std::vector<int32_t>& request_ids,
                                          void* executor_token) {
  if (request_ids.empty()) { return; }
  const Backend backend = GetUniqueBackend(job_id, request_ids);
  backends_.at(backend)->ExecuteRequests(job_id, request_ids, executor_token);
}

Backend ExecutorImpl::GetUniqueBackend(const int64_t job_id,
                                       const std::vector<int32_t>& request_ids) {
  const Backend backend =
      request_store_->MutRequestEntry(job_id, request_ids.front())->desc().op_desc().backend();
  for (int32_t i = 1; i < request_ids.size(); ++i) {
    CHECK_EQ(request_store_->MutRequestEntry(job_id, request_ids.at(i))->desc().op_desc().backend(),
             backend);
  }
  return backend;
}

struct Scheduler::Impl {
  Impl();
  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<Executor> executor;
  std::shared_ptr<Coordinator> coordinator;
};

Scheduler::Impl::Impl() {
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  request_store.reset(new RequestStore());
  executor.reset(new ExecutorImpl());
  executor->Init(request_store);
  if (conf.has_static_group_coordinator_conf()
      || conf.coordinator_conf_case() == CollectiveBoxingConf::COORDINATOR_CONF_NOT_SET) {
    coordinator.reset(new StaticGroupCoordinator());
  } else if (conf.has_dynamic_coordinator_conf()) {
#ifdef WITH_MPI
    coordinator.reset(new DynamicCoordinator());
#else
    LOG(FATAL) << "MPI components not found";
#endif
  } else {
    UNIMPLEMENTED();
  }
  coordinator->Init(request_store, executor);
}

std::shared_ptr<const CollectiveBoxingExecutorPlanToken> Scheduler::AddPlan(const Plan& plan) {
  std::vector<int64_t> job_ids;
  for (const auto& job_id7request_set : plan.collective_boxing_plan().job_id2request_set()) {
    const int64_t job_id = job_id7request_set.first;
    job_ids.push_back(job_id);
  }
  LOG(INFO) << "AddPlan " << plan.collective_boxing_plan().DebugString();
  impl_->request_store->AddPlan(plan.collective_boxing_plan());
  impl_->request_store->DebugLog();
  impl_->executor->AddPlan(job_ids);
  impl_->coordinator->AddPlan(job_ids);
  return std::make_shared<CollectiveBoxingExecutorPlanToken>(job_ids);
}

void Scheduler::DeletePlan(
    const std::shared_ptr<const CollectiveBoxingExecutorPlanToken> plan_token) {
  const std::vector<int64_t>& job_ids = plan_token->job_ids();
  impl_->coordinator->DeletePlan(job_ids);
  impl_->executor->DeletePlan(job_ids);
  impl_->request_store->DeletePlan(job_ids);
  LOG(INFO) << "Scheduler DeletePlan";
  impl_->request_store->DebugLog();
}

Scheduler::Scheduler() { impl_.reset(new Impl()); }

Scheduler::~Scheduler() = default;

std::shared_ptr<RequestHandle> Scheduler::CreateRequestHandle(const RankDesc& rank_desc) {
  const std::pair<int64_t, int32_t> pair =
      impl_->request_store->GetJobId7RequestIdByName(rank_desc.op_desc().name());
  const int64_t job_id = pair.first;
  const int32_t request_id = pair.second;
  auto* request_entry = impl_->request_store->MutRequestEntry(job_id, request_id);
  CHECK(rank_desc.op_desc() == request_entry->desc().op_desc());
  const int32_t local_rank = request_entry->GlobalRankToLocalRank(rank_desc.rank());
  void* request_entry_token = impl_->request_store->CreateRequestEntryToken(job_id, request_id);
  void* request_token = impl_->coordinator->CreateRequestToken(job_id, request_id);
  void* executor_token = impl_->executor->CreateExecutorToken(job_id, request_id);
  return std::make_shared<RequestHandle>(job_id, request_id, local_rank, request_entry_token,
                                         request_token, executor_token);
}

void Scheduler::Schedule(const std::shared_ptr<RequestHandle>& handle,
                         std::shared_ptr<const RuntimeRequestInfo> request_info) {
  const int32_t job_id = handle->job_id();
  const int32_t request_id = handle->request_id();
  const int32_t local_rank = handle->local_rank();
  const bool ready = impl_->request_store->GetRequestEntry(handle->request_entry_token())
                         ->AddRuntimeRequest(local_rank, std::move(request_info));
  if (ready) { impl_->coordinator->AddRequest(handle->request_token(), handle->executor_token()); }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
