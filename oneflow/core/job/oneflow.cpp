#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/improver.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/profiler.h"
#include "oneflow/core/job/sub_plan.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/critical_section_desc.h"
#include "oneflow/core/job/available_memory_desc.pb.h"
#include "oneflow/core/job/foreign_job_instance.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/graph/plan_task_graph.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/model_init_job.h"
#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/operator/interface_op_util.h"

DECLARE_bool(grpc_use_no_signal);

namespace std {

template<>
struct hash<oneflow::ParallelBlobConf> {
  size_t operator()(const oneflow::ParallelBlobConf& parallel_blob_conf) const {
    std::string serialized;
    parallel_blob_conf.SerializeToString(&serialized);
    return std::hash<std::string>()(serialized);
  }
};

}  // namespace std

namespace oneflow {

bool operator==(const ParallelBlobConf& lhs, const ParallelBlobConf& rhs) {
  return BlobDesc(lhs.logical_blob_desc_conf()) == BlobDesc(rhs.logical_blob_desc_conf())
         && lhs.parallel_conf() == rhs.parallel_conf() && lhs.sbp_conf() == rhs.sbp_conf();
}

namespace {

std::string cluster_thrd_ids_key(const std::string& plan_name) {
  return plan_name + "_cluster_thrd_ids";
}

std::string net_topo_key(const std::string& plan_name) { return plan_name + "_net_topo"; }

std::string sub_plan_key(const std::string& plan_name, int64_t machine_id, int64_t thrd_id) {
  return plan_name + "_" + std::to_string(machine_id) + "_" + std::to_string(thrd_id);
}

std::string total_mbn_num_key(const std::string& plan_name) { return plan_name + "_total_mbn_num"; }

void PushPlan(const std::string& plan_name, const Plan& plan) {
  HashMap<int64_t, std::set<int64_t>> machine_id2thrd_id_set;
  HashMap<std::pair<int64_t, int64_t>, std::vector<TaskProto>> mchn_thrd_id2task_protos;
  for (const auto& task : plan.task()) {
    machine_id2thrd_id_set[task.machine_id()].insert(task.thrd_id());
    mchn_thrd_id2task_protos[std::make_pair(task.machine_id(), task.thrd_id())].emplace_back(task);
  }

  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  for (const auto& pair : machine_id2thrd_id_set) {
    CHECK(machine_id2thrd_ids.emplace(pair.first, ThrdIds()).second);
    std::vector<int64_t> thrd_id_vec(pair.second.begin(), pair.second.end());
    *(machine_id2thrd_ids.at(pair.first).mutable_thrd_id()) = StdVec2PbRf(thrd_id_vec);
  }

  ClusterThrdIds cluster_thrd_ids;
  *(cluster_thrd_ids.mutable_machine_id2thrd_ids()) = HashMap2PbMap(machine_id2thrd_ids);
  Global<CtrlClient>::Get()->PushKV(cluster_thrd_ids_key(plan_name), cluster_thrd_ids);

  for (const auto& pair : mchn_thrd_id2task_protos) {
    SubPlan sub_plan;
    *(sub_plan.mutable_task()) = StdVec2PbRpf(pair.second);
    Global<CtrlClient>::Get()->PushKV(sub_plan_key(plan_name, pair.first.first, pair.first.second),
                                      sub_plan);
  }
  Global<CtrlClient>::Get()->PushKV(total_mbn_num_key(plan_name),
                                    std::to_string(plan.total_mbn_num()));

  Global<CtrlClient>::Get()->PushKV(net_topo_key(plan_name), plan.net_topo());
}

void PullPlan(const std::string& plan_name, Plan* plan) {
  ClusterThrdIds cluster_thrd_ids;
  Global<CtrlClient>::Get()->PullKV(cluster_thrd_ids_key(plan_name), &cluster_thrd_ids);
  PrintProtoToTextFile(cluster_thrd_ids, JoinPath(FLAGS_log_dir, cluster_thrd_ids_key(plan_name)));
  HashMap<int64_t, ThrdIds> machine_id2thrd_ids;
  machine_id2thrd_ids = PbMap2HashMap(cluster_thrd_ids.machine_id2thrd_ids());
  int64_t machine_id = Global<MachineCtx>::Get()->this_machine_id();
  auto thrd_ids_it = machine_id2thrd_ids.find(machine_id);
  CHECK(thrd_ids_it != machine_id2thrd_ids.end());
  std::vector<int64_t> thrd_id_vec = PbRf2StdVec(thrd_ids_it->second.thrd_id());
  for (auto thrd_id : thrd_id_vec) {
    SubPlan sub_plan;
    Global<CtrlClient>::Get()->PullKV(sub_plan_key(plan_name, machine_id, thrd_id), &sub_plan);
    plan->mutable_task()->MergeFrom(sub_plan.task());
  }
  NetTopo net_topo;
  std::string total_mbn_num;
  Global<CtrlClient>::Get()->PullKV(total_mbn_num_key(plan_name), &total_mbn_num);
  plan->set_total_mbn_num(oneflow_cast<int64_t>(total_mbn_num));
  Global<CtrlClient>::Get()->PullKV(net_topo_key(plan_name), &net_topo);
  *(plan->mutable_net_topo()) = net_topo;
}

void CompileCurJobOnMaster(Job* job, Plan* improved_plan, bool need_job_complete) {
  const JobDesc& job_desc = GlobalJobDesc();
  Plan naive_plan;
  Plan mem_shared_plan;
  double start = GetCurTime();
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    Compiler().Compile(job, &naive_plan, need_job_complete);
    LOG(INFO) << "compile time: " << GetCurTime() - start;
    mem_shared_plan =
        Improver().ImproveMemSharedIdOnly(*Global<AvailableMemDesc>::Get(), naive_plan);
    OF_BARRIER();
    TeePersistentLogStream::Create("naive_plan")->Write(naive_plan);
    TeePersistentLogStream::Create("mem_shared_plan")->Write(mem_shared_plan);
    LOG(INFO) << "push_pull_plan:" << GetCurTime() - start;
  }
  if (job_desc.enable_experiment_run()) {
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      PushPlan("mem_shared_plan", mem_shared_plan);
    } else {
      PullPlan("mem_shared_plan", &mem_shared_plan);
    }
    // Experiment Runtime
    { Runtime experiment_run(mem_shared_plan, job_desc.piece_num_of_experiment_phase(), true); }
    // Improve
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      TeePersistentLogStream::Create("available_mem_desc")->Write(*Global<AvailableMemDesc>::Get());
      CHECK_GT(Global<AvailableMemDesc>::Get()->machine_amd_size(), 0);
      *improved_plan = Improver().Improve(
          *Global<AvailableMemDesc>::Get(), naive_plan,
          JoinPath(FLAGS_log_dir, ActEventLogger::experiment_act_event_bin_filename()));
      OF_BARRIER();
      TeePersistentLogStream::Create("improved_plan")->Write(*improved_plan);
      Global<CtrlClient>::Get()->Clear();
      OF_BARRIER();
    }
  } else {
    *improved_plan = mem_shared_plan;
  }
  LOG(INFO) << "compile and improve time: " << GetCurTime() - start;
}

size_t ComputeTotalPieceNum() {
  const auto& job_descs = *Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  size_t total_piece_num = 0;
  for (const auto& job_desc : job_descs) {
    total_piece_num = std::max<size_t>(total_piece_num,
                                       job_desc->NumOfPiecesInBatch() * job_desc->TotalBatchNum());
  }
  return total_piece_num;
}

void MergePlan(Plan* plan, const std::vector<Plan>& sub_plans) {
  CHECK(!sub_plans.empty());
  *plan = sub_plans.at(0);
  FOR_RANGE(int32_t, i, 1, sub_plans.size()) {
    plan->mutable_task()->MergeFrom(sub_plans.at(i).task());
    plan->set_total_mbn_num(plan->total_mbn_num() + sub_plans.at(i).total_mbn_num());
  }
  Compiler().GenNetTopo(plan);
}

RegstDescProto* GetSoleDataRegstDescProto(TaskProto* task) {
  RegstDescProto* ret = nullptr;
  for (auto& pair : *task->mutable_produced_regst_desc()) {
    CHECK(pair.second.regst_desc_type().has_data_regst_desc());
    CHECK_ISNULL(ret);
    ret = &pair.second;
  }
  CHECK_NOTNULL(ret);
  return ret;
}

const OperatorConf& GetSoleOpConf(const TaskProto& task) {
  CHECK_EQ(task.exec_sequence().exec_node_size(), 1);
  return task.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
}

void UpdateSoleObnRegstDescId(TaskProto* task) {
  CHECK_EQ(task->exec_sequence().exec_node_size(), 1);
  auto* exec_node = task->mutable_exec_sequence()->mutable_exec_node(0);
  const auto& obns = exec_node->kernel_conf().op_attribute().output_bns();
  CHECK_EQ(obns.size(), 1);
  int64_t regst_desc_id = GetSoleDataRegstDescProto(task)->regst_desc_id();
  (*exec_node->mutable_bn_in_op2regst_desc_id())[obns.Get(0)] = regst_desc_id;
}

// example
// given caller plan: op_A --> op_identity_tick --> op_B
// given callee plan: op_src_tick --> op_C --> op_D --> op_E --> op_sink_tick
// return:
//         op_A --> op_identity_tick --> op_C --> op_D --> op_E --> op_sink_tick --> op_B
//                                        /
//                        op_src_tick ---/
//
// note: after this function called, op_src_tick is illegal and need to be deleted from plan
void LinkTickTaskProto(TaskProto* identity_tick, TaskProto* src_tick, TaskProto* sink_tick) {
  CHECK(GetSoleOpConf(*identity_tick).has_tick_conf());
  CHECK(GetSoleOpConf(*src_tick).has_source_tick_conf());
  CHECK(GetSoleOpConf(*sink_tick).has_sink_tick_conf());
  RegstDescProto* id_tick_sole_regst = GetSoleDataRegstDescProto(identity_tick);
  RegstDescProto* src_tick_sole_regst = GetSoleDataRegstDescProto(src_tick);
  RegstDescProto* sink_tick_sole_regst = GetSoleDataRegstDescProto(sink_tick);

  sink_tick_sole_regst->set_regst_desc_id(id_tick_sole_regst->regst_desc_id());
  *sink_tick_sole_regst->mutable_consumer_task_id() = id_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(sink_tick);

  id_tick_sole_regst->set_regst_desc_id(src_tick_sole_regst->regst_desc_id());
  *id_tick_sole_regst->mutable_consumer_task_id() = src_tick_sole_regst->consumer_task_id();
  UpdateSoleObnRegstDescId(identity_tick);
}

void LinkMainPlan(Plan* plan, const Plan& main_plan,
                  const std::vector<std::string>& identity_tick_op_names) {
  std::function<bool(const TaskProto*)> IsInterfaceTickTockTask;
  {
    auto task_ids = std::make_shared<HashSet<int64_t>>();
    for (const auto& task : main_plan.task()) {
      if (task.task_type() == TaskType::kTick) { CHECK(task_ids->emplace(task.task_id()).second); }
    }
    IsInterfaceTickTockTask = [task_ids](const TaskProto* task) {
      if (task_ids->find(task->task_id()) != task_ids->end()) { return true; }
      if (task->exec_sequence().exec_node_size() != 1) { return false; }
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      OperatorConf::OpTypeCase op_type_case = kernel_conf.op_attribute().op_conf().op_type_case();
      return op_type_case == OperatorConf::kSourceTickConf
             || op_type_case == OperatorConf::kSinkTickConf;
    };
  }
  plan->mutable_task()->MergeFrom(main_plan.task());
  HashMap<std::string, TaskProto*> sole_tick_op_name2sole_task;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (IsInterfaceTickTockTask(task) == false) { continue; }
    const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
    const auto& op_name = kernel_conf.op_attribute().op_conf().name();
    CHECK(sole_tick_op_name2sole_task.emplace(op_name, task).second);
  }
  FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    const CriticalSection& critical_section =
        Global<CriticalSectionDesc>::Get()->GetCriticalSection(i);
    LinkTickTaskProto(sole_tick_op_name2sole_task.at(identity_tick_op_names.at(i)),
                      sole_tick_op_name2sole_task.at(critical_section.source_tick_op_name()),
                      sole_tick_op_name2sole_task.at(critical_section.sink_tick_op_name()));
  }
  {
    // erase source_tick task_proto
    HashSet<std::string> source_tick_op_names;
    FOR_RANGE(int32_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      const CriticalSection& critical_section =
          Global<CriticalSectionDesc>::Get()->GetCriticalSection(i);
      CHECK(source_tick_op_names.emplace(critical_section.source_tick_op_name()).second);
    }
    Erase<PbRpf<TaskProto>>(*plan->mutable_task(), [&](const TaskProto& task) {
      if (task.task_type() == TaskType::kSourceTick) {
        CHECK(task.exec_sequence().exec_node_size() == 1);
        const auto& kernel_conf = task.exec_sequence().exec_node(0).kernel_conf();
        const auto& op_conf = kernel_conf.op_attribute().op_conf();
        CHECK(op_conf.has_source_tick_conf());
        CHECK(source_tick_op_names.find(op_conf.name()) != source_tick_op_names.end());
        return true;
      } else {
        return false;
      }
    });
  }
}

void GetMemSharingOpBlobInfo(const JobBuilder& job_builder, const std::string& op_name,
                             ParallelBlobConf* blob_conf) {
  std::string obn = "out";
  std::string lbn;
  {
    const auto& op_conf = job_builder.OpConf4OpName(op_name);
    if (op_conf.has_variable_conf()) {
      lbn = op_name + "/" + op_conf.variable_conf().out();
    } else if (op_conf.has_input_conf()) {
      lbn = op_name + "/" + op_conf.input_conf().out();
    } else if (op_conf.has_output_conf()) {
      lbn = op_name + "/" + op_conf.output_conf().out();
    } else if (op_conf.has_switch_output_conf()) {
      lbn = op_name + "/" + op_conf.switch_output_conf().out();
    } else if (op_conf.has_return_conf()) {
      lbn = op_name + "/" + op_conf.return_conf().out();
    } else {
      UNIMPLEMENTED();
    }
  }
  const auto& job = job_builder.job();
  ParallelBlobConf ret;
  *blob_conf->mutable_parallel_conf() = job_builder.ParallelConf4OpName(op_name);
  *blob_conf->mutable_logical_blob_desc_conf() = job.helper().lbn2logical_blob_desc().at(lbn);
  *blob_conf->mutable_sbp_conf() =
      job.sbp_conf().op_name2sbp_signature_conf().at(op_name).bn_in_op2sbp_parallel().at(obn);
  LogicalBlobId lbi(GenLogicalBlobId(lbn));
  const auto& lbis = job.helper().batch_dim_lbis();
  blob_conf->set_has_batch_dim(std::find(lbis.begin(), lbis.end(), lbi) != lbis.end());
}

void FilterOpName2ParallelBlobConf(
    const HashSet<OperatorConf::OpTypeCase>& match, std::vector<Job>* jobs,
    HashMap<std::string, ParallelBlobConf>* op_name2parallel_blob_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs->size()) {
    JobBuilder job_builder(&jobs->at(job_id));
    for (const OperatorConf& op_conf : jobs->at(job_id).net().op()) {
      if (match.find(op_conf.op_type_case()) == match.end()) { continue; }
      const auto& iter = op_name2parallel_blob_conf->find(op_conf.name());
      if (iter == op_name2parallel_blob_conf->end()) {
        auto* first_op_parallel_blob_conf = &(*op_name2parallel_blob_conf)[op_conf.name()];
        GetMemSharingOpBlobInfo(job_builder, op_conf.name(), first_op_parallel_blob_conf);
      } else {
        ParallelBlobConf parallel_blob_conf;
        GetMemSharingOpBlobInfo(job_builder, op_conf.name(), &parallel_blob_conf);
        CHECK(parallel_blob_conf == iter->second);
      }
    }
  }
}

void FilterArgPassJobGroupInfo(
    std::vector<Job>* jobs,
    HashMap<ParallelBlobConf, HashMap<std::string, std::vector<std::string>>>*
        parallel_blob_conf2input_op_name2output_op_name) {
  HashMap<ParallelBlobConf, HashSet<std::string>> parallel_blob_conf2input_op_names;
  HashMap<ParallelBlobConf, HashSet<std::string>> parallel_blob_conf2output_op_names;
  FOR_RANGE(int64_t, job_id, 0, jobs->size()) {
    JobBuilder job_builder(&jobs->at(job_id));
    for (const OperatorConf& op_conf : jobs->at(job_id).net().op()) {
      if (IsInterfaceOpConf(op_conf) == false) { continue; }
      ParallelBlobConf parallel_blob_conf;
      GetMemSharingOpBlobInfo(job_builder, op_conf.name(), &parallel_blob_conf);
      if (op_conf.has_input_conf()) {
        parallel_blob_conf2input_op_names[parallel_blob_conf].insert(op_conf.name());
      }
      if (op_conf.has_return_conf()) {
        parallel_blob_conf2output_op_names[parallel_blob_conf].insert(op_conf.name());
      }
    }
  }
  for (const auto& pair : parallel_blob_conf2input_op_names) {
    const auto& parallel_blob_conf = pair.first;
    for (const auto& input_op_name : pair.second) {
      const auto& output_op_names = parallel_blob_conf2output_op_names[parallel_blob_conf];
      if (output_op_names.empty()) { continue; }
      for (const auto& output_op_name : output_op_names) {
        if (input_op_name == output_op_name) { continue; }
        auto* in2outs = &(*parallel_blob_conf2input_op_name2output_op_name)[parallel_blob_conf];
        (*in2outs)[input_op_name].push_back(output_op_name);
      }
    }
  }
}

void MakeMainJob(const std::vector<Job>& jobs, Job* main_job,
                 std::vector<std::string>* identity_tick_op_names,
                 LogicalBlobId* critical_section_sink_lbi) {
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  std::vector<OperatorConf> op_confs;
  OperatorConf wait_and_send_ids_op_conf;
  {
    wait_and_send_ids_op_conf.set_name(std::string("System-Main-WaitAndSendIds_") + NewUniqueId());
    auto* wait_and_send_ids_conf = wait_and_send_ids_op_conf.mutable_wait_and_send_ids_conf();
    wait_and_send_ids_conf->set_out("out");
    wait_and_send_ids_conf->set_wait_buffer_name(kBufferNameGlobalWaitJobId);
    wait_and_send_ids_conf->set_data_type(DataType::kInt32);
    FOR_RANGE(int64_t, i, 0, Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->size()) {
      const auto& cs_idx = Global<CriticalSectionDesc>::Get()->CriticalSectionIds4JobId(i);
      *wait_and_send_ids_conf->add_id_list()->mutable_id() = {cs_idx.begin(), cs_idx.end()};
    }
  }
  op_confs.push_back(wait_and_send_ids_op_conf);
  OperatorConf reentrant_lock_op_conf;
  {
    reentrant_lock_op_conf.set_name(std::string("System-Main-ReentrantLock_") + NewUniqueId());
    auto* reentrant_lock_conf = reentrant_lock_op_conf.mutable_reentrant_lock_conf();
    reentrant_lock_conf->set_start(wait_and_send_ids_op_conf.name() + "/out");
    // ibn "end" is set after plan generated because we don't like cycle in job
    reentrant_lock_conf->set_out("out");
    Global<CriticalSectionDesc>::Get()->DumpCriticalSectionId2IntersectinIds(
        reentrant_lock_conf->mutable_lock_id2intersecting_lock_ids());
  }
  op_confs.push_back(reentrant_lock_op_conf);
  // critical section case op conf
  OperatorConf cs_case_op_conf;
  {
    cs_case_op_conf.set_name(std::string("System-Main-Case_") + NewUniqueId());
    auto* cs_case_conf = cs_case_op_conf.mutable_case_conf();
    cs_case_conf->set_in(reentrant_lock_op_conf.name() + "/out");
    FOR_RANGE(int64_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
      cs_case_conf->add_out(GenRepeatedBn("out", i));
    }
  }
  op_confs.push_back(cs_case_op_conf);
  FOR_RANGE(int64_t, i, 0, Global<CriticalSectionDesc>::Get()->CriticalSectionNum()) {
    OperatorConf identity_tick_op_conf;
    std::string name_prefix = "System-Main-Tick_CriticalSection_";
    identity_tick_op_conf.set_name(name_prefix + std::to_string(i));
    auto* identity_tick_conf = identity_tick_op_conf.mutable_tick_conf();
    identity_tick_conf->add_tick(cs_case_op_conf.name() + "/" + GenRepeatedBn("out", i));
    identity_tick_conf->set_out("out");
    identity_tick_op_names->push_back(identity_tick_op_conf.name());
    op_confs.push_back(identity_tick_op_conf);
  }
  // critical section esac op conf
  OperatorConf cs_esac_op_conf;
  {
    cs_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    auto* cs_esac_conf = cs_esac_op_conf.mutable_esac_conf();
    for (const auto& identity_tick_op_name : *identity_tick_op_names) {
      cs_esac_conf->add_in(identity_tick_op_name + "/out");
    }
    cs_esac_conf->set_out("out");
    cs_esac_conf->set_data_type(DataType::kInt32);
  }
  op_confs.push_back(cs_esac_op_conf);
  OperatorConf callback_notify_esac_op_conf;
  {
    callback_notify_esac_op_conf.set_name(std::string("System-Main-Esac_") + NewUniqueId());
    auto* callback_notify_esac_conf = callback_notify_esac_op_conf.mutable_esac_conf();
    for (int64_t total_job_cs_id :
         Global<CriticalSectionDesc>::Get()->job_id2total_job_critical_section_id()) {
      callback_notify_esac_conf->add_in(identity_tick_op_names->at(total_job_cs_id) + "/out");
    }
    callback_notify_esac_conf->set_out("out");
    callback_notify_esac_conf->set_data_type(DataType::kInt32);
  }
  op_confs.push_back(callback_notify_esac_op_conf);
  OperatorConf callback_notify_op_conf;
  {
    callback_notify_op_conf.set_name(std::string("System-Main-CallbackNotify_") + NewUniqueId());
    auto* callback_notify_conf = callback_notify_op_conf.mutable_callback_notify_conf();
    callback_notify_conf->set_in(callback_notify_esac_op_conf.name() + "/out");
    FOR_RANGE(int64_t, i, 0,
              Global<CriticalSectionDesc>::Get()->job_id2total_job_critical_section_id().size()) {
      const auto& buffer_name = GetCallbackNotifierBufferName(GlobalJobDesc(i).job_name());
      callback_notify_conf->add_callback_buffer_name(buffer_name);
    }
  }
  op_confs.push_back(callback_notify_op_conf);

  critical_section_sink_lbi->set_op_name(cs_esac_op_conf.name());
  critical_section_sink_lbi->set_blob_name("out");

  ParallelConf parallel_conf;
  parallel_conf.set_policy(kDataParallel);
  parallel_conf.add_device_name("0:cpu:0");
  JobBuilder(main_job).AddOps(parallel_conf, op_confs);
  auto* job_conf = main_job->mutable_job_conf();
  job_conf->set_job_name("MainJob-unamed");
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
  job_conf->set_default_data_type(DataType::kInt32);
}

void ConnectCriticalSectionEndToReentrantLockEnd(Plan* main_plan,
                                                 const LogicalBlobId& critical_section_sink_lbi) {
  TaskProto* reentrant_lock_task = nullptr;
  TaskProto* cs_sink_task = nullptr;
  FOR_RANGE(int64_t, i, 0, main_plan->task_size()) {
    auto* task = main_plan->mutable_task(i);
    CHECK_EQ(task->exec_sequence().exec_node_size(), 1);
    if (task->task_type() == TaskType::kReentrantLock) {
      CHECK_ISNULL(reentrant_lock_task);
      reentrant_lock_task = task;
    } else {
      const auto& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      if (critical_section_sink_lbi.op_name() == kernel_conf.op_attribute().op_conf().name()) {
        CHECK_ISNULL(cs_sink_task);
        cs_sink_task = task;
      }
    }
  }
  CHECK_NOTNULL(reentrant_lock_task);
  CHECK_NOTNULL(cs_sink_task);
  RegstDescProto* cs_end_regst = PlanUtil::GetSoleProducedDataRegst(cs_sink_task);
  cs_end_regst->add_consumer_task_id(reentrant_lock_task->task_id());
  reentrant_lock_task->mutable_consumed_regst_desc_id()->at("in").add_regst_desc_id(
      cs_end_regst->regst_desc_id());

  auto* reentrant_exec_node = reentrant_lock_task->mutable_exec_sequence()->mutable_exec_node(0);
  (*reentrant_exec_node->mutable_bn_in_op2regst_desc_id())["end"] = cs_end_regst->regst_desc_id();

  auto* op_attribute = reentrant_exec_node->mutable_kernel_conf()->mutable_op_attribute();
  op_attribute->add_input_bns("end");
  (*op_attribute->mutable_bn_in_op2lbi())["end"] = critical_section_sink_lbi;

  auto* reentrant_lock_conf = op_attribute->mutable_op_conf()->mutable_reentrant_lock_conf();
  reentrant_lock_conf->set_end(GenLogicalBlobName(critical_section_sink_lbi));
}

void CompileMainJob(Job* main_job, const LogicalBlobId& critical_section_sink_lbi, int32_t job_id,
                    Plan* main_plan) {
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  WithJobIdGlobal(job_id, [&]() { CompileCurJobOnMaster(main_job, main_plan, false); });
  ConnectCriticalSectionEndToReentrantLockEnd(main_plan, critical_section_sink_lbi);
}

void AddGlobalJobDesc(const Job& job, int32_t job_id) {
  auto* job_descs = Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  CHECK_EQ(job_descs->size(), job_id);
  job_descs->emplace_back(new JobDesc(job, job_id));
  CHECK(Global<JobName2JobId>::Get()->emplace(job.job_conf().job_name(), job_id).second);
}

bool NeedAllocateMemory(const RegstDescTypeProto& regst_desc_type) {
  return regst_desc_type.has_data_regst_desc()
         && regst_desc_type.data_regst_desc().packed_blob_desc().is_body_disabled() == false;
}

void FinishGlobalCriticalSectionDesc(const std::vector<Plan>& plans) {
  std::vector<HashMap<std::string, HashSet<int64_t>>> job_id2sole_op_name2mem_block_ids;
  std::vector<HashSet<int64_t>> job_id2mem_block_ids;
  for (const auto& plan : plans) {
    job_id2sole_op_name2mem_block_ids.push_back(HashMap<std::string, HashSet<int64_t>>());
    auto* sole_op_name2mem_block_ids = &job_id2sole_op_name2mem_block_ids.back();
    job_id2mem_block_ids.push_back(HashSet<int64_t>());
    auto* job_id_mem_block_ids = &job_id2mem_block_ids.back();
    for (const auto& task : plan.task()) {
      if (task.exec_sequence().exec_node_size() == 1) {
        const auto& kernel_conf = task.exec_sequence().exec_node(0).kernel_conf();
        const auto& op_name = kernel_conf.op_attribute().op_conf().name();
        auto* mem_block_ids = &(*sole_op_name2mem_block_ids)[op_name];
        for (const auto& pair : task.produced_regst_desc()) {
          if (NeedAllocateMemory(pair.second.regst_desc_type())) {
            mem_block_ids->emplace(pair.second.mem_shared_id());
          }
        }
      }
      for (const auto& pair : task.produced_regst_desc()) {
        if (NeedAllocateMemory(pair.second.regst_desc_type())) {
          job_id_mem_block_ids->emplace(pair.second.mem_shared_id());
        }
      }
    }
  }
  HashMap<int64_t, HashSet<int64_t>> job_id2input_output_mem_block_ids;
  auto* critical_section_desc = Global<CriticalSectionDesc>::Get();
  // set mem_block_id for InputOutputCriticalSection
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSection(i);
    int64_t job_id = critical_section->job_id();
    auto* input_output_mem_block_ids = &job_id2input_output_mem_block_ids[job_id];
    if (critical_section->has_input_output_critical_section()) {
      HashSet<int64_t> mem_block_ids;
      for (const auto& op_name :
           critical_section->input_output_critical_section().lbi_producer_op_name()) {
        const auto& cur_mem_block_ids = job_id2sole_op_name2mem_block_ids.at(job_id).at(op_name);
        mem_block_ids.insert(cur_mem_block_ids.begin(), cur_mem_block_ids.end());
      }
      *critical_section->mutable_mem_block_id() = {mem_block_ids.begin(), mem_block_ids.end()};
      input_output_mem_block_ids->insert(mem_block_ids.begin(), mem_block_ids.end());
    } else {
      CHECK(critical_section->has_total_job_critical_section());
    }
  }
  HashSet<int64_t> unique_job_id_check;
  // set mem_block_id for TotalJobCriticalSection
  FOR_RANGE(int64_t, i, 0, critical_section_desc->CriticalSectionNum()) {
    auto* critical_section = critical_section_desc->MutCriticalSection(i);
    int64_t job_id = critical_section->job_id();
    const auto& input_output_mem_block_ids = job_id2input_output_mem_block_ids.at(job_id);
    if (critical_section->has_total_job_critical_section()) {
      CHECK(unique_job_id_check.emplace(job_id).second);
      auto* mem_block_ids = &job_id2mem_block_ids.at(job_id);
      {
        // exclude input/output criticalsection mem_blob_ids from total_job
        auto it = mem_block_ids->begin();
        while (it != mem_block_ids->end()) {
          if (input_output_mem_block_ids.find(*it) == input_output_mem_block_ids.end()) {
            ++it;
          } else {
            it = mem_block_ids->erase(it);
          }
        }
      }
      *critical_section->mutable_mem_block_id() = {mem_block_ids->begin(), mem_block_ids->end()};
    }
  }
  critical_section_desc->Done();
}

void MakePullJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_output_or_var_op_name2pull_job_name();
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name;
  DataType data_type;
  JobBuilder job_builder(job);
  OperatorConf input_op_conf;
  {
    input_op_conf.set_name(op_name);
    auto* input_conf = input_op_conf.mutable_input_conf();
    input_conf->set_out("out");
    auto* blob_conf = input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
    data_type = blob_conf->data_type();
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {input_op_conf});
  }
  OperatorConf foreign_output_op_conf;
  {
    foreign_output_op_conf.set_name(std::string("System-Pull-ForeignOutput_") + NewUniqueId());
    auto* foreign_output_conf = foreign_output_op_conf.mutable_foreign_output_conf();
    foreign_output_conf->set_in(input_op_conf.name() + "/out");
    foreign_output_conf->set_ofblob_buffer_name(GetForeignOutputBufferName(job_name));
    ParallelConf parallel_conf;
    parallel_conf.set_policy(kDataParallel);
    parallel_conf.add_device_name("0:cpu:0");
    job_builder.AddOps(parallel_conf, {foreign_output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->add_arg_op_name(input_op_conf.name());
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
  job_conf->set_default_data_type(data_type);
}

void MakePushJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_input_or_var_op_name2push_job_name();
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name;
  DataType data_type;
  JobBuilder job_builder(job);
  OperatorConf foreign_input_op_conf;
  {
    foreign_input_op_conf.set_name(std::string("System-Push-ForeignInput_") + NewUniqueId());
    auto* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
    foreign_input_conf->set_out("out");
    foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
    auto* blob_conf = foreign_input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
    data_type = blob_conf->data_type();
    ParallelConf parallel_conf;
    parallel_conf.set_policy(kDataParallel);
    parallel_conf.add_device_name("0:cpu:0");
    job_builder.AddOps(parallel_conf, {foreign_input_op_conf});
  }
  OperatorConf output_op_conf;
  {
    output_op_conf.set_name(op_name);
    auto* output_conf = output_op_conf.mutable_output_conf();
    output_conf->set_in(foreign_input_op_conf.name() + "/out");
    output_conf->set_out("out");
    InterfaceOpUtil::InitBlobConf(output_conf->mutable_blob_conf(), parallel_blob_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->add_arg_op_name(output_op_conf.name());
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
  job_conf->set_default_data_type(data_type);
}

void MakeArgPassJob(const std::string& job_name, const ParallelBlobConf& parallel_blob_conf,
                    const std::string& input_op_name,
                    const std::vector<std::string>& output_op_names, Job* job) {
  CHECK_EQ(output_op_names.empty(), false);
  for (const auto& output_op_name : output_op_names) { CHECK_NE(output_op_name, input_op_name); }
  auto* op_name2arg_pass_job_info =
      Global<InterUserJobInfo>::Get()->mutable_input_or_var_op_name2arg_pass_job_info();
  CHECK(op_name2arg_pass_job_info->find(input_op_name) == op_name2arg_pass_job_info->end());
  auto* arg_pass_job_info = &(*op_name2arg_pass_job_info)[input_op_name];
  arg_pass_job_info->set_intput_or_var_op_name(input_op_name);
  arg_pass_job_info->set_arg_pass_job_name(job_name);
  auto* op_name2in_index = arg_pass_job_info->mutable_output_or_var_op_name2in_index();

  JobBuilder job_builder(job);
  OperatorConf foreign_input_op_conf;
  {
    foreign_input_op_conf.set_name(std::string("System-ArgPass-ForeignInput_") + NewUniqueId());
    auto* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
    foreign_input_conf->set_out("out");
    foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
    auto* blob_conf = foreign_input_conf->mutable_blob_conf();
    blob_conf->mutable_shape()->add_dim(1);
    blob_conf->set_data_type(DataType::kInt32);
    blob_conf->set_has_dim0_valid_num(false);
    blob_conf->set_has_dim1_valid_num(false);
    blob_conf->set_has_dim2_valid_num(false);
    blob_conf->set_split_axis(0);
    blob_conf->set_has_batch_dim(false);
    ParallelConf parallel_conf;
    parallel_conf.set_policy(kDataParallel);
    parallel_conf.add_device_name("0:cpu:0");
    job_builder.AddOps(parallel_conf, {foreign_input_op_conf});
  }
  std::vector<OperatorConf> input_op_confs(output_op_names.size());
  FOR_RANGE(int64_t, i, 0, output_op_names.size()) {
    input_op_confs.at(i).set_name(output_op_names.at(i));
    (*op_name2in_index)[output_op_names.at(i)] = i;
    auto* input_conf = input_op_confs.at(i).mutable_input_conf();
    input_conf->set_out("out");
    auto* blob_conf = input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
  }
  job_builder.AddOps(parallel_blob_conf.parallel_conf(), input_op_confs);
  OperatorConf switch_output_op_conf;
  {
    switch_output_op_conf.set_name(input_op_name);
    auto* switch_output_conf = switch_output_op_conf.mutable_switch_output_conf();
    switch_output_conf->set_in_index(foreign_input_op_conf.name() + "/out");
    for (const auto& op_conf : input_op_confs) {
      switch_output_conf->add_in(op_conf.name() + "/out");
    }
    switch_output_conf->set_out("out");
    InterfaceOpUtil::InitBlobConf(switch_output_conf->mutable_blob_conf(), parallel_blob_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {switch_output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  *job_conf->mutable_arg_op_name() = {output_op_names.begin(), output_op_names.end()};
  job_conf->add_arg_op_name(input_op_name);
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
}

void MakeModelSaveJob(const std::string& job_name,
                      const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf,
                      Job* job) {
  Global<InterUserJobInfo>::Get()->set_global_model_save_job_name(job_name);
  JobBuilder job_builder(job);
  ParallelConf md_save_parallel_conf;
  // only save on master
  md_save_parallel_conf.add_device_name("0:cpu:0");
  md_save_parallel_conf.set_policy(ParallelPolicy::kDataParallel);
  {
    // there is always a tick op in model save job, so we can ignore the case with no variable
    OperatorConf tick_op_conf;
    tick_op_conf.set_name("System-Saver-tick");
    tick_op_conf.mutable_tick_conf()->set_out("out");
    job_builder.AddOps(md_save_parallel_conf, {tick_op_conf});
  }
  for (const auto& pair : var_op_name2parallel_blob_conf) {
    const auto& var_op_name = pair.first;
    OperatorConf input_op_conf;
    {
      const auto& parallel_blob_conf = pair.second;
      input_op_conf.set_name(var_op_name);
      auto* input_conf = input_op_conf.mutable_input_conf();
      input_conf->set_out("out");
      auto* blob_conf = input_conf->mutable_blob_conf();
      InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
      CHECK(blob_conf->has_data_type());
      job_builder.AddOps(parallel_blob_conf.parallel_conf(), {input_op_conf});
    }
    OperatorConf model_save_op_conf;
    {
      model_save_op_conf.set_name("System-Saver-" + var_op_name + "-MdSave");
      ModelSaveV2OpConf* model_save_conf = model_save_op_conf.mutable_model_save_v2_conf();
      model_save_conf->set_in(input_op_conf.name() + "/out");
      model_save_conf->set_lbn(input_op_conf.name() + "/out");
      job_builder.AddOps(md_save_parallel_conf, {model_save_op_conf});
    }
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_piece_size(1);
  job_conf->set_data_part_num(1);
  job_conf->set_total_batch_num(1);
}

void CompileAndMergePlanOnMaster(const PbRpf<Job>& conf_jobs, Plan* plan) {
  std::vector<Job> jobs(conf_jobs.size());
  std::vector<Plan> sub_plans(conf_jobs.size());
  FOR_RANGE(int32_t, i, 0, sub_plans.size()) {
    jobs.at(i) = conf_jobs.Get(i);
    WithJobIdGlobal(i, [&]() { CompileCurJobOnMaster(&jobs.at(i), &sub_plans.at(i), true); });
  }
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    // only has user job in jobs and sub_plans in this time
    InterJobMemSharingUtil::MergeMemBlockBetweenSubPlans(jobs, &sub_plans);
  }
  HashMap<std::string, ParallelBlobConf> push_op_name2parallel_blob_conf;
  FilterOpName2ParallelBlobConf({OperatorConf::kInputConf}, &jobs,
                                &push_op_name2parallel_blob_conf);
  HashMap<std::string, ParallelBlobConf> pull_op_name2parallel_blob_conf;
  FilterOpName2ParallelBlobConf({OperatorConf::kReturnConf}, &jobs,
                                &pull_op_name2parallel_blob_conf);
  HashMap<std::string, ParallelBlobConf> var_op_name2parallel_blob_conf;
  FilterOpName2ParallelBlobConf({OperatorConf::kVariableConf}, &jobs,
                                &var_op_name2parallel_blob_conf);
  HashMap<ParallelBlobConf, HashMap<std::string, std::vector<std::string>>>
      parallel_blob_conf2input_op_name2output_op_name;
  FilterArgPassJobGroupInfo(&jobs, &parallel_blob_conf2input_op_name2output_op_name);
  int64_t job_id = -1;
  {
    size_t helper_job_size =
        push_op_name2parallel_blob_conf.size() + pull_op_name2parallel_blob_conf.size();

    for (const auto& pair : parallel_blob_conf2input_op_name2output_op_name) {
      helper_job_size += pair.second.size();
    }
    // + 2 for model init job and model save job
    helper_job_size += 2;
    size_t user_job_size = jobs.size();
    jobs.resize(user_job_size + helper_job_size);
    sub_plans.resize(user_job_size + helper_job_size);
    job_id = user_job_size;
  }
  auto CompileHelperJob = [&](Job* job) {
    jobs.at(job_id) = *job;
    AddGlobalJobDesc(*job, job_id);
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      WithJobIdGlobal(job_id, [&]() { CompileCurJobOnMaster(job, &sub_plans.at(job_id), true); });
    }
    ++job_id;
  };
  for (const auto& pair : push_op_name2parallel_blob_conf) {
    Job push_job;
    MakePushJob(std::string("System-Push-") + pair.first, pair.first, pair.second, &push_job);
    CompileHelperJob(&push_job);
  }
  for (const auto& pair : pull_op_name2parallel_blob_conf) {
    Job pull_job;
    MakePullJob(std::string("System-Pull-") + pair.first, pair.first, pair.second, &pull_job);
    CompileHelperJob(&pull_job);
  }
  for (const auto& outer_pair : parallel_blob_conf2input_op_name2output_op_name) {
    const auto parallel_blob_conf = outer_pair.first;
    for (const auto& pair : outer_pair.second) {
      Job arg_pass_job;
      MakeArgPassJob("System-ArgPass-" + pair.first, parallel_blob_conf, pair.first, pair.second,
                     &arg_pass_job);
      CompileHelperJob(&arg_pass_job);
    }
  }
  {
    Job model_init_job;
    HashMap<std::string, OperatorConf> var_op_name2op_conf;
    FilterVariableOps(jobs, &var_op_name2op_conf);
    MakeModelInitJob("System-ModelInit", &model_init_job, var_op_name2op_conf,
                     var_op_name2parallel_blob_conf);
    CompileHelperJob(&model_init_job);
  }
  {
    Job model_save_job;
    MakeModelSaveJob("System-ModelSave", var_op_name2parallel_blob_conf, &model_save_job);
    CompileHelperJob(&model_save_job);
  }
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    InterJobMemSharingUtil::BindInterfaceMemBlockId(jobs, &sub_plans);
    FinishGlobalCriticalSectionDesc(sub_plans);
    MergePlan(plan, sub_plans);
    Plan main_plan;
    std::vector<std::string> identity_tick_op_names;
    {
      Job main_job;
      LogicalBlobId critical_section_sink_lbi;
      MakeMainJob(jobs, &main_job, &identity_tick_op_names, &critical_section_sink_lbi);
      AddGlobalJobDesc(main_job, job_id);
      CompileMainJob(&main_job, critical_section_sink_lbi, sub_plans.size(), &main_plan);
    }
    LinkMainPlan(plan, main_plan, identity_tick_op_names);
    TeePersistentLogStream::Create("merged_plan")->Write(*plan);
    PushPlan("merged_plan", *plan);
  } else {
    PullPlan("merged_plan", plan);
  }
}

}  // namespace

GlobalObjectsScope4JobConf::GlobalObjectsScope4JobConf(const JobSet& job_set) {
  Global<const JobMemSharingStrategy>::New(job_set.job_mem_sharing_strategy());
  Global<JobName2JobId>::New();
  Global<std::vector<std::unique_ptr<JobDesc>>>::New();
  FOR_RANGE(int32_t, job_id, 0, job_set.job_size()) {
    auto* job_desc = new JobDesc(job_set.job(job_id), job_id);
    Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->emplace_back(job_desc);
    CHECK(Global<JobName2JobId>::Get()->emplace(job_desc->job_name(), job_desc->job_id()).second);
  }
}

GlobalObjectsScope4JobConf::~GlobalObjectsScope4JobConf() {
  Global<JobName2JobId>::Delete();
  Global<std::vector<std::unique_ptr<JobDesc>>>::Delete();
  Global<const JobMemSharingStrategy>::Delete();
}

Oneflow::Oneflow(const oneflow::JobSet& job_set) {
  global_objects_scope4job_conf_.reset(new GlobalObjectsScope4JobConf(job_set));
  // Runtime
  CompileAndMergePlanOnMaster(job_set.job(), &plan_);
  if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
    PushPlan("plan", plan_);
  } else {
    PullPlan("plan", &plan_);
  }
  runtime_.reset(new Runtime(plan_, ComputeTotalPieceNum(), false));
}

Oneflow::~Oneflow() {
  runtime_.reset();
  if (Global<Profiler>::Get() != nullptr) {
    Global<Profiler>::Get()->Profile(
        plan_, JoinPath(FLAGS_log_dir, ActEventLogger::act_event_bin_filename()));
  }
  global_objects_scope4job_conf_.reset();
}

}  // namespace oneflow

int main(int argc, char** argv) {
  UNIMPLEMENTED();
  return 0;
}
