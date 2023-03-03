// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/*
For the op to work you need to run the `OpToIdentityPattern` after autodiffing the op.
*/

#include <map>
#include <memory>
#include <snap/Tensor.hpp>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/alias/aliasmodel.hpp>
#include <popart/basicoptionals.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/popx/op/collectives/replicatedallreducex.hpp>
#pragma GCC diagnostic pop


#include <poplar/Graph.hpp>
#include <poputil/exceptions.hpp>

#include <gcl/Collectives.hpp>

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart
{
  // -------------- Op --------------
  class ReplicatedAllReduceTPOp : public ReplicatedAllReduceOp
  {
  public:
    ReplicatedAllReduceTPOp(
        const OperatorIdentifier &_opid,
        const CollectiveOperator &op_,
        const ReplicaGrouping &grouping_,
        const bool backwards_,
        const Op::Settings &settings_)
        : ReplicatedAllReduceOp(_opid, op_, grouping_, settings_),
          grouping(grouping_),
          backwards(backwards_) {}

    std::unique_ptr<Op> clone() const override
    {
      return std::make_unique<ReplicatedAllReduceTPOp>(*this);
    }

    void setup() override
    {
      if (op != CollectiveOperator::Add)
      {
        throw error(
            "Cannot create ReplicatedAllReduceTPOp op. "
            "CollectiveOperator::Add is the only collective operator "
            "that is currently implemented.");
      }
      ReplicatedAllReduceOp::setup();
    }

    std::vector<std::unique_ptr<Op>> getGradOps() override
    {
      std::vector<std::unique_ptr<Op>> result;
      result.push_back(std::make_unique<ReplicatedAllReduceTPOp>(
          opid, op, grouping, !backwards, settings));
      return result;
    }

    void appendOutlineAttributes(OpSerialiserBase &os) const override
    {
      Op::appendOutlineAttributes(os);
      os.appendAttribute("op", op);
      os.appendAttribute("backwards", backwards);
    }

    // Replace the op with an idenity depending on which pass it operates in
    bool canBeReplacedByIdentity() const override
    {
      bool isInference = getIr().getExecutionMode() == Ir::ExecutionMode::Inference;
      return backwards && (getIr().hasConstructedBackwards() || isInference);
    }

    const std::vector<GradInOutMapper> &gradInputInfo() const override
    {
      static const std::vector<GradInOutMapper> inInfo = {
          {getInIndex(), getOutIndex(), GradOpInType::GradOut}};
      return inInfo;
    }

    const std::map<int, int> &gradOutToNonGradIn() const override
    {
      static const std::map<int, int> outInfo = {{getOutIndex(), getInIndex()}};
      return outInfo;
    }

    static OperatorIdentifier defaultOperatorId()
    {
      return OperatorIdentifier{
          "ai.graphcore",
          "ReplicatedAllReduceTP",
          1,      // Op version
          {1, 1}, // number of inputs
          1       // number of outputs
      };
    }

  protected:
    ReplicaGrouping grouping;
    bool backwards = false;
  };

  const popart::OperatorIdentifier ReplicatedAllReduceTP =
      ReplicatedAllReduceTPOp::defaultOperatorId();

  static OpDefinition::DataTypes T = {
      DataType::FLOAT, DataType::FLOAT16, DataType::INT32, DataType::UINT32};

  static OpDefinition ReplicatedAllReduceTPOpDef(
      {OpDefinition::Inputs({{"X", T}}),
       OpDefinition::Outputs({{"Y", T}}),
       OpDefinition::Attributes(
           {{sCollectiveOperator, {"*"}},
            {sCollectiveReplicaGrouping, {"*"}},
            {"backwards", {"*"}}})});

  static OpCreator<ReplicatedAllReduceTPOp> ReplicatedAllReduceTPOpCreator(
      OpDefinitions({{ReplicatedAllReduceTP, ReplicatedAllReduceTPOpDef}}),
      [](const OpCreatorInfo &info)
      {
        CollectiveOperator op = static_cast<CollectiveOperator>(
            info.attributes.getAttribute<Attributes::Int>(
                sCollectiveOperator, static_cast<int>(CollectiveOperator::Add)));
        const auto grouping = extractReplicaGroupingFromAttrs(
            info.attributes,
            info.settings.getIr().getSessionOptions().getGlobalReplicationFactor());
        return std::make_unique<ReplicatedAllReduceTPOp>(
            info.opid,
            op,
            grouping,
            info.attributes.getAttribute<Attributes::Int>("backwards"),
            info.settings);
      },
      true);

  // -------------- OpX --------------
  namespace popx
  {

    class ReplicatedAllReduceTPOpx : public ReplicatedAllReduceOpx
    {
    public:
      ReplicatedAllReduceTPOpx(Op *op, Devicex *devicex)
          : ReplicatedAllReduceOpx(op, devicex)
      {
        verifyOp<ReplicatedAllReduceTPOp>(
            op, {ReplicatedAllReduceTPOp::defaultOperatorId()});
        if (op_p->canBeReplacedByIdentity())
        {
          throw error(
              "You need to run the `OpToIdentityPattern` pattern before "
              "running the IR.");
        }
      }
    };

    popx::OpxCreator<ReplicatedAllReduceTPOpx>
        ReplicatedAllReduceTPOpxCreator(ReplicatedAllReduceTP);
  } // namespace popx
} // namespace popart
