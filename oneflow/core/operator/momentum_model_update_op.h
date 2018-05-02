#ifndef ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

class MomentumModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumModelUpdateOp);
  MomentumModelUpdateOp() = default;
  ~MomentumModelUpdateOp() = default;

  void InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void MdUpdtVirtualInitFromOpConf() override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
