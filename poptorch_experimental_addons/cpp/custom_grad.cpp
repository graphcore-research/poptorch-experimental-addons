// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/alias/aliasmodel.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensorindex.hpp>
#pragma GCC diagnostic pop

namespace {

struct GradOp : popart::Op {
    static const popart::OperatorIdentifier ID;
    explicit GradOp(const popart::OperatorIdentifier& opid_, const popart::Op::Settings& settings_)
        : popart::Op(opid_, settings_) {}

    std::unique_ptr<popart::Op> clone() const final { return std::make_unique<GradOp>(*this); }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    void setup() final { outInfo(0) = inInfo(0); }
    // Grad
    const std::vector<popart::GradInOutMapper>& gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> info = {
            {0, 0, popart::GradOpInType::GradOut}};
        return info;
    }
    const std::map<int, int>& gradOutToNonGradIn() const {
        static const std::map<int, int> info = {{0, 1}};
        return info;
    }
    // Aliasing metadata
    popart::view::Regions aliases(popart::InIndex, popart::OutIndex) const final {
        return {popart::view::Region::getFull(inShape(0))};
    }
    void growAliasModel(popart::AliasModel& aliasModel) const final {
        aliasModel.insertUnaryModifier(*this, 0);
    }
    popart::view::RegMap fwdRegMap(popart::InIndex, popart::OutIndex) const {
        return [](const popart::view::Region& region) -> popart::view::Regions { return {region}; };
    }
    popart::view::RegMap bwdRegMap(popart::InIndex in, popart::OutIndex out) const {
        return fwdRegMap(in, out);  // identity mapping is its self-inverse
    }
};

struct Op : popart::Op {
    static const popart::OperatorIdentifier ID;
    explicit Op(const popart::OperatorIdentifier& opid_, const popart::Op::Settings& settings_)
        : popart::Op(opid_, settings_) {}

    std::unique_ptr<popart::Op> clone() const final { return std::make_unique<Op>(*this); }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    void setup() final { outInfo(0) = inInfo(0); }
    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<popart::Op>> gradOps;
        gradOps.emplace_back(new GradOp(GradOp::ID, settings));
        return gradOps;
    }
    // Aliasing metadata
    popart::view::Regions aliases(popart::InIndex in, popart::OutIndex) const final {
        if (in != 0) return {};
        return {popart::view::Region::getFull(inShape(0))};
    }
    void growAliasModel(popart::AliasModel& aliasModel) const final {
        aliasModel.insertUnaryModifier(*this, 0);
    }
    popart::view::RegMap fwdRegMap(popart::InIndex in, popart::OutIndex) const {
        assert(in == 0 && "fwdRegMap called for non-aliased input");
        return [](const popart::view::Region& region) -> popart::view::Regions { return {region}; };
    }
    popart::view::RegMap bwdRegMap(popart::InIndex in, popart::OutIndex out) const {
        return fwdRegMap(in, out);  // identity mapping is its self-inverse
    }
};

// Note that we can reuse the Opx for the grad, as both just set `output = inputs[0]`.
struct Opx : popart::popx::Opx {
    Opx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<popart::Op>(op, {Op::ID, GradOp::ID});
    }
    void grow(poplar::program::Sequence&) const final {
        if (hasInput(1)) {
            popart::logging::warn(
                "`custom_gradient(f, g)` has not pruned the forward pass of `g`, leading to"
                " inefficient execution - please use the setting:"
                " `PopTorchOptions._popart.setPatterns(dict(CustomGradientOpPatten=True))`");
        }
        insert(outId(0), get(inId(0)));
    }
};

const popart::OperatorIdentifier Op::ID = {"ai.graphcore.pea", "CustomGradient", 1};
const popart::OperatorIdentifier GradOp::ID = {"ai.graphcore.pea", "CustomGradientGrad", 1};
popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16, popart::DataType::FLOAT};
popart::OpCreator<Op> opCreator(
    {{Op::ID,
      {popart::OpDefinition::Inputs({{"f", T}, {"g", T}}),
       popart::OpDefinition::Outputs({{"output", T}}), popart::OpDefinition::Attributes()}}},
    [](const popart::OpCreatorInfo& info) {
        return std::make_unique<Op>(info.opid, info.settings);
    },
    true);
popart::popx::OpxCreator<Opx> opxCreator({Op::ID, GradOp::ID});

// CustomGradientOpPatten

struct CustomGradientOpPatten : popart::PreAliasPattern {
    bool matches(popart::Op* op) const override { return op->opid == Op::ID; }

    std::vector<const popart::Tensor*> touches(popart::Op* op) const override {
        std::vector<const popart::Tensor*> touches;
        if (op->hasInput(1)) {
            return {op->input->tensor(1)};
        }
        return {};
    }

    bool apply(popart::Op* op) const override {
        if (op->getIr().hasConstructedBackwards() && op->hasInput(1)) {
            op->disconnectInTensor(1);
            return true;
        }
        return false;
    }
};

static popart::PatternCreator<CustomGradientOpPatten> RemoveAllReducePatternCreator(
    "CustomGradientOpPatten",
    false);

}  // namespace
