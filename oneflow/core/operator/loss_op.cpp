#include "oneflow/core/operator/loss_op.h"

namespace oneflow {

namespace {

class LossSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossSbpSignature);
  ~LossSbpSignature() override = default;

  LossSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), S(0)) -> S(0)";
  }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& ibn : op().input_bns()) {
      (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0);
    }
    for (const auto& obn : op().output_bns()) {
      (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0);
    }
    (*bn2sbp)["loss_instance_num"].mutable_partial_sum_parallel();
    if (!op().GetValFromCustomizedConf<std::string>("weight").empty()) {
      (*bn2sbp)["reduction_coefficient"].mutable_partial_sum_parallel();
    }
  }
};

}  // namespace
void LossOp::InitFromOpConf() {
  EnrollInputBn("prediction");
  if (HasFieldInCustomizedConf("label")) { EnrollInputBn("label", false); }
  EnrollOutputBn("loss", false);
  EnrollOutputBn("loss_instance_num", false);
  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    EnrollInputBn("weight", false);
    EnrollOutputBn("reduction_coefficient", false);
  }

  VirtualInitFromOpConf();
}

void LossOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LossKernelConf* conf = GetMutLossKernelConf(kernel_conf);
  conf->set_prediction_type(GetBlobDesc4BnInOp("prediction")->data_type());
  if (HasFieldInCustomizedConf("label")) {
    conf->set_label_type(GetBlobDesc4BnInOp("label")->data_type());
  } else {
    conf->set_label_type(DataType::kInvalidDataType);
  }
  conf->set_weight_scalar(GetValFromCustomizedConf<float>("weight_scalar"));
  conf->set_reduction(static_cast<LossReductionType>(GetEnumFromCustomizedConf("reduction")));
}

void LossOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  if (HasFieldInCustomizedConf("label")) {
    const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
    CHECK_EQ(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
    CHECK_EQ(pred_blob_desc->has_dim0_valid_num_field(),
             label_blob_desc->has_dim0_valid_num_field());
    CHECK_EQ(pred_blob_desc->has_dim0_inner_shape(), label_blob_desc->has_dim0_inner_shape());
  }
  if (pred_blob_desc->has_dim0_inner_shape()) {
    CHECK_EQ(pred_blob_desc->dim0_inner_shape().At(0), 1);
  }
  CHECK_GT(pred_blob_desc->shape().NumAxes(), 0);
  // loss
  BlobDesc* loss_blob_desc = GetBlobDesc4BnInOp("loss");
  *loss_blob_desc = *pred_blob_desc;
  loss_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
  loss_blob_desc->set_data_type(pred_blob_desc->data_type());
  // loss instance num
  BlobDesc* loss_instance_num_blob_desc = GetBlobDesc4BnInOp("loss_instance_num");
  loss_instance_num_blob_desc->mut_shape() = Shape({1});
  loss_instance_num_blob_desc->set_data_type(pred_blob_desc->data_type());
  loss_instance_num_blob_desc->set_has_data_id_field(pred_blob_desc->has_data_id_field());

  if (!GetValFromCustomizedConf<std::string>("weight").empty()) {
    // reduction_coefficient
    BlobDesc* reduction_blob_desc = GetBlobDesc4BnInOp("reduction_coefficient");
    reduction_blob_desc->mut_shape() = Shape({1});
    reduction_blob_desc->set_data_type(pred_blob_desc->data_type());
  }
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

void LossOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new LossSbpSignature(this));
}

LogicalBlobId LossOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  if (output_bn == "loss_instance_num") {
    ret.set_blob_name("loss_instance_num");
  } else if (output_bn == "reduction_coefficient") {
    ret.set_blob_name("reduction_coefficient");
  } else {
    ret.set_blob_name(GetValFromCustomizedConf<std::string>(output_bn));
  }
  return ret;
}

}  // namespace oneflow
