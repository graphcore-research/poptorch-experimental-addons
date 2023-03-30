// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#ifndef GUARD_NEURALNET_REPLICATEDALLTOALL_HPP
#define GUARD_NEURALNET_REPLICATEDALLTOALL_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/op/collectives/collectives.hpp>
#pragma GCC diagnostic pop

namespace popart {

class ReplicatedAllToAllOp : public CollectivesBaseOp {
   public:
    ReplicatedAllToAllOp(const OperatorIdentifier&, ReplicaGrouping group, const Op::Settings&);

    std::unique_ptr<Op> clone() const final;
    void setup() final;

    float getSubgraphValue() const final { return getHighSubgraphValue(); }

    ReplicatedTensorShardingIndices getReplicatedTensorShardingIndices() const override;

    std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReplicatedAllToAllGradOp : public ReplicatedAllToAllOp {
   public:
    ReplicatedAllToAllGradOp(const ReplicatedAllToAllOp&);

    const std::map<int, int>& gradOutToNonGradIn() const override final;
    const std::vector<GradInOutMapper>& gradInputInfo() const final;
};

}  // namespace popart

#endif
