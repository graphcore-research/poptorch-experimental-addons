#ifndef GUARD_NEURALNET_REPLICATEDALLTOALLX_HPP
#define GUARD_NEURALNET_REPLICATEDALLTOALLX_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/popx/op/collectives/collectivesx.hpp>
#pragma GCC diagnostic pop

namespace popart {
namespace popx {

class ReplicatedAllToAllOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllToAllOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class ReplicatedAllToAllGradOpx : public ReplicatedAllToAllOpx {
public:
  ReplicatedAllToAllGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif
