// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
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

ReplicatedAllToAllOp::ReplicatedAllToAllOp(const OperatorIdentifier& _opid,
                                           ReplicaGrouping group,
                                           const Op::Settings& settings_)
    : CollectivesBaseOp(_opid, group, settings_) {}

std::unique_ptr<Op> ReplicatedAllToAllOp::clone() const {
    return std::make_unique<ReplicatedAllToAllOp>(*this);
}

void ReplicatedAllToAllOp::setup() {
    outInfo(getOutIndex()) = inInfo(getInIndex());
}

static OpDefinition::DataTypes T = {DataType::FLOAT, DataType::FLOAT16, DataType::INT32,
                                    DataType::UINT32};

static OpDefinition ReplicatedAllToAllOpDef({OpDefinition::Inputs({{"X", T}}),
                                             OpDefinition::Outputs({{"Y", T}}),
                                             OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllToAllOp> ReplicatedAllToAllOpCreator(
    OpDefinitions({{{"ai.graphcore", "ReplicatedAllToAll", 1}, ReplicatedAllToAllOpDef}}),
    [](const OpCreatorInfo& info) {
        return std::make_unique<ReplicatedAllToAllOp>(
            info.opid,
            extractReplicaGroupingFromAttrs(
                info.attributes,
                info.settings.getIr().getSessionOptions().getGlobalReplicationFactor()),
            info.settings);
    },
    true);

ReplicatedTensorShardingIndices ReplicatedAllToAllOp::getReplicatedTensorShardingIndices() const {
    return {{{ReplicatedAllToAllOp::getInIndex()}, {}}};
}

std::vector<std::unique_ptr<popart::Op>> ReplicatedAllToAllOp::getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(std::make_unique<ReplicatedAllToAllGradOp>(*this));
    return upops;
}

ReplicatedAllToAllGradOp::ReplicatedAllToAllGradOp(const ReplicatedAllToAllOp& op)
    : ReplicatedAllToAllOp(op.opid, op.getReplicaGrouping(), op.settings) {}

const std::map<int, int>& ReplicatedAllToAllGradOp::gradOutToNonGradIn() const {
    static const std::map<int, int> outInfo = {{getOutIndex(), ReplicatedAllToAllOp::getInIndex()}};
    return outInfo;
}

const std::vector<GradInOutMapper>& ReplicatedAllToAllGradOp::gradInputInfo() const {
    static const std::vector<GradInOutMapper> inInfo = {
        {getInIndex(), ReplicatedAllToAllOp::getOutIndex(), GradOpInType::GradOut}};
    return inInfo;
}

namespace popx {

class ReplicatedAllToAllOpx : public CollectivesBaseOpx {
   public:
    ReplicatedAllToAllOpx(Op*, Devicex*);
    void grow(poplar::program::Sequence&) const final;
    InputCreatorType getInputCreatorType(InIndex index) const final;
    poplar::Tensor unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
    view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class ReplicatedAllToAllGradOpx : public ReplicatedAllToAllOpx {
   public:
    ReplicatedAllToAllGradOpx(Op*, Devicex*);
};

ReplicatedAllToAllOpx::ReplicatedAllToAllOpx(Op* op, Devicex* devicex)
    : CollectivesBaseOpx(op, devicex) {
    verifyOp<ReplicatedAllToAllOp>(op, {"ai.graphcore", "ReplicatedAllToAll", 1});
}

void ReplicatedAllToAllOpx::grow(poplar::program::Sequence& prog) const {
    auto& op = getOp<ReplicatedAllToAllOp>();

    const poplar::OptionFlags& allToAllOptions = dv_p->lowering().gclOptions;

    poplar::Tensor output =
        gcl::allToAllCrossReplica(graph(), getInTensor(ReplicatedAllToAllOp::getInIndex()), prog,
                                  toGclCommGroup(op.getReplicaGrouping()),
                                  debugContext("replicatedAllToAll"), allToAllOptions);

    setOutTensor(ReplicatedAllToAllOp::getOutIndex(), output);
}

InputCreatorType ReplicatedAllToAllOpx::getInputCreatorType(InIndex) const {
    return InputCreatorType::CanUnwind;
}

poplar::Tensor ReplicatedAllToAllOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                         InIndex,
                                                         OutIndex) const {
    return tensor;
}

view::RegMap ReplicatedAllToAllOpx::unwindRegion(InIndex, OutIndex) const {
    return [](const view::Region& r) { return view::Regions(1, r); };
}

ReplicatedAllToAllGradOpx::ReplicatedAllToAllGradOpx(Op* op, Devicex* devicex)
    : ReplicatedAllToAllOpx(op, devicex) {
    verifyOp<ReplicatedAllToAllGradOp>(op, {"ai.graphcore", "ReplicatedAllToAllGrad", 1});
}

namespace {
OpxCreator<ReplicatedAllToAllOpx> ReplicatedAllToAllOpxCreator({"ai.graphcore",
                                                                "ReplicatedAllToAll", 1});

OpxCreator<ReplicatedAllToAllGradOpx> ReplicatedAllToAllGradOpxCreator({"ai.graphcore",
                                                                        "ReplicatedAllToAllGrad",
                                                                        1});
}  // namespace
}  // namespace popx
}  // namespace popart
