// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cmath>
#include <memory>

#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#pragma GCC diagnostic pop

namespace {

poplar::Tensor quantise(poplar::Graph& graph,
                        const poplar::Tensor& input,
                        unsigned exponentBits,
                        unsigned mantissaBits,
                        bool stochasticRounding,
                        poplar::program::Sequence& prog,
                        const poplar::DebugContext& debugContext) {
    auto halfInput = input;
    if (input.elementType() != poplar::HALF) {
        halfInput = popops::cast(graph, input, poplar::HALF, prog, {debugContext, "toHalf"});
    }
    int maxExponent = (1 << (exponentBits - 1)) - 1;
    auto absMax = static_cast<float>(std::pow(2.0, maxExponent) *
                                     (2 - std::pow(2.0, -static_cast<double>(mantissaBits))));
    // std::max accounts for E5M2 which has a larger range (smaller min normal) than FP16
    // (this means we probably lose some values in this range)
    auto downscale = std::max(1.0f, static_cast<float>(std::pow(2.0, 14 - maxExponent)));
    auto mask = static_cast<unsigned short>((1u << (10 - mantissaBits)) - 1);

    auto cs = graph.addComputeSet(debugContext);
    auto tileMapping = graph.getTileMapping(halfInput);
    auto output = graph.clone(halfInput);
    for (auto tile = 0u; tile < tileMapping.size(); ++tile) {
        auto tileAbsMax = graph.addConstant<float>(poplar::HALF, {}, absMax);
        auto tileDownscale = graph.addConstant<float>(poplar::HALF, {}, downscale);
        auto tileMask = graph.addConstant<unsigned short>(poplar::UNSIGNED_SHORT, {}, mask);
        graph.setTileMapping(tileAbsMax, tile);
        graph.setTileMapping(tileDownscale, tile);
        graph.setTileMapping(tileMask, tile);
        for (auto& interval : tileMapping[tile]) {
            auto out = output.flatten().slice(interval);
            auto vertexName = stochasticRounding ? "CustomQuant<RoundingMode::Stochastic>"
                                                 : "CustomQuant<RoundingMode::Nearest>";
            graph.setTileMapping(graph.addVertex(cs, vertexName,
                                                 {{"input", halfInput.flatten().slice(interval)},
                                                  {"output", out},
                                                  {"absMax", tileAbsMax},
                                                  {"downscale", tileDownscale},
                                                  {"mask", tileMask}}),
                                 tile);
        }
    }
    prog.add(poplar::program::Execute(cs, debugContext));
    if (input.elementType() != poplar::HALF) {
        return popops::cast(graph, output, poplar::FLOAT, prog, {debugContext, "toFloat"});
    }
    return output;
}

struct Config {
    std::string rootPath;
    unsigned exponentBits;
    unsigned mantissaBits;
    std::string rounding;
    bool fwd;
    std::string bwd;  // "quantise|ste|stop"
};

struct Op : popart::Op {
    static const popart::OperatorIdentifier ID;
    Config config;

    explicit Op(const popart::OperatorIdentifier& opid_,
                const popart::Op::Settings& settings_,
                const Config& config)
        : popart::Op(opid_, settings_), config(config) {}

    // Basics
    std::unique_ptr<popart::Op> clone() const final { return std::make_unique<Op>(*this); }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    void setup() final { outInfo(0) = inInfo(0); }
    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<popart::Op>> gradOps;
        if (config.bwd != "stop") {
            auto gradConfig = config;
            gradConfig.fwd = (config.bwd == "quantise");
            gradOps.emplace_back(new Op(ID, settings, gradConfig));
        }
        return gradOps;
    }
    bool requiresRandomSeed() const override { return true; }
    // Grad properties
    const std::vector<popart::GradInOutMapper>& gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> info = {
            {0, 0, popart::GradOpInType::GradOut}};
        return info;
    }
    const std::map<int, int>& gradOutToNonGradIn() const {
        static const std::map<int, int> info = {{0, 0}};
        return info;
    }
    // Attributes for hashing
    void appendAttributes(popart::OpSerialiserBase& os) const final {
        popart::Op::appendAttributes(os);
        appendAttributesImpl(os);
    }
    void appendOutlineAttributes(popart::OpSerialiserBase& os) const final {
        popart::Op::appendOutlineAttributes(os);
        appendAttributesImpl(os);
    }

   private:
    void appendAttributesImpl(popart::OpSerialiserBase& os) const {
        os.appendAttribute("rootPath", config.rootPath);
        os.appendAttribute("exponentBits", config.exponentBits);
        os.appendAttribute("mantissaBits", config.mantissaBits);
        os.appendAttribute("rounding", config.rounding);
        os.appendAttribute("fwd", config.fwd);
        os.appendAttribute("bwd", config.bwd);
    }
};

// Note that we can reuse the Opx for the grad.
struct Opx : popart::popx::Opx {
    Opx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<Op>(op, Op::ID);
        graph().addCodelets(getOp<Op>().config.rootPath + "/cpp/simulated_quant_codelet.cpp");
    }
    void grow(poplar::program::Sequence& prog) const final {
        auto input = get(inId(0));
        auto& op = getOp<Op>();
        if (op.config.fwd) {
            insert(outId(0),
                   quantise(graph(), input, op.config.exponentBits, op.config.mantissaBits,
                            op.config.rounding == "stochastic", prog, debugContext()));
        } else {
            // Since we never declare input/output aliases, we must create a copy
            auto output = graph().clone(input);
            prog.add(poplar::program::Copy(input, output, /*dontOutline*/ false, debugContext()));
            insert(outId(0), output);
        }
    }
};

const popart::OperatorIdentifier Op::ID = {"ai.graphcore", "SimulatedQuant", 1};
popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT, popart::DataType::FLOAT16};
popart::OpCreator<Op> opCreator(
    {{Op::ID,
      {popart::OpDefinition::Inputs({{"input", T}}), popart::OpDefinition::Outputs({{"output", T}}),
       popart::OpDefinition::Attributes({{"root_path", {"string"}},
                                         {"exponent_bits", {"int"}},
                                         {"mantissa_bits", {"int"}},
                                         {"rounding", {"string"}},
                                         {"fwd", {"int"}},
                                         {"bwd", {"string"}}})}}},
    [](const popart::OpCreatorInfo& info) {
        auto rootPath = info.attributes.getAttribute<popart::Attributes::String>("root_path");
        auto exponentBits =
            unsigned(info.attributes.getAttribute<popart::Attributes::Int>("exponent_bits"));
        auto mantissaBits =
            unsigned(info.attributes.getAttribute<popart::Attributes::Int>("mantissa_bits"));
        auto rounding = info.attributes.getAttribute<popart::Attributes::String>("rounding");
        auto fwd = bool(info.attributes.getAttribute<popart::Attributes::Int>("fwd"));
        auto bwd = info.attributes.getAttribute<popart::Attributes::String>("bwd");
        return std::make_unique<Op>(
            info.opid, info.settings,
            Config{rootPath, exponentBits, mantissaBits, rounding, fwd, bwd});
    },
    true);
popart::popx::OpxCreator<Opx> opxCreator({Op::ID});

}  // namespace
