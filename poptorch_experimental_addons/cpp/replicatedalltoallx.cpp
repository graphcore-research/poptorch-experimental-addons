#include <gcl/Collectives.hpp>

#include "replicatedalltoall.hpp"
#include "replicatedalltoallx.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#pragma GCC diagnostic pop

namespace popart {
namespace popx {

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
