// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <cmath>
#include <sstream>

#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

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

void mapTensor2Dblocks(poplar::Graph& graph, poplar::Tensor& t) {
    assert(t.rank() == 2 && "only 2D tensors can use mapTensor2Dblocks");
    auto nTiles = graph.getTarget().getNumTiles();
    auto blockSize0 =
        std::max<unsigned>(std::ceil(t.dim(0) / nTiles),
                           std::ceil(std::sqrt(static_cast<float>(t.numElements()) / nTiles)));
    auto nBlocks0 = (t.dim(0) + blockSize0 - 1) / blockSize0;
    auto nBlocks1 = std::max<unsigned>(1u, nTiles / nBlocks0);
    auto blockSize1 = (t.dim(1) + nBlocks1 - 1) / nBlocks1;
    for (auto i = 0u; i < nBlocks0; ++i) {
        for (auto j = 0u; j < nBlocks1; ++j) {
            auto tile = nBlocks1 * i + j;
            graph.setTileMapping(t.slice({std::min<unsigned>(i * blockSize0, t.dim(0)),
                                          std::min<unsigned>(j * blockSize1, t.dim(1))},
                                         {std::min<unsigned>((i + 1) * blockSize0, t.dim(0)),
                                          std::min<unsigned>((j + 1) * blockSize1, t.dim(1))}),
                                 tile);
        }
    }
}

poplar::Tensor getCachedCopy(std::map<std::pair<size_t, size_t>, poplar::Tensor>& cache,
                             poplar::Graph& graph,
                             size_t tile,
                             size_t index,
                             const poplar::Tensor& t,
                             poplar::program::Sequence& prog) {
    std::pair<size_t, size_t> key = {tile, index};
    auto iter = cache.find(key);
    if (iter != cache.end()) {
        return iter->second;
    }
    poplar::Tensor copy = graph.addVariable(t.elementType(), t.shape());
    graph.setTileMapping(copy, tile);
    cache[key] = copy;
    prog.add(poplar::program::Copy(t, copy));
    return copy;
}

poplar::Tensor l1distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        std::ostringstream msg;
        msg << "Bad arguments to l1distance, expected a.shape (M, K), b.shape (N, "
               "K), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t n = b.dim(0);
    poplar::Tensor out =
        graph.addVariable(a.elementType(), {a.dim(0), b.dim(0)}, {debugContext, "l1dist_out"});
    mapTensor2Dblocks(graph, out);
    const auto& mapping = graph.getTileMapping(out);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l1dist"});
    const auto vertexName = poputil::templateVertex("L1DistanceSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a_index = j / n;
                size_t b_index = j % n;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a_index]);
                graph.connect(v["b"], b[b_index]);
                graph.connect(v["out"], out[a_index][b_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return out;
}

poplar::Tensor l1distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || gradOutput.rank() != 2 || a.dim(1) != b.dim(1) ||
        gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0)) {
        std::ostringstream msg;
        msg << "Bad arguments to l1distancegrad, expected"
            << " a.shape (M, K), b.shape (N, K), gradOutput.shape (M, N), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString()
            << ", gradOutput.shape = " << gradOutput.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t k = a.dim(1);
    poplar::Tensor grad =
        graph.addVariable(a.elementType(), a.shape(), {debugContext, "l1dist_grad"});
    mapTensor2Dblocks(graph, grad);
    const auto& mapping = graph.getTileMapping(grad);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l1dist_grad"});
    const auto vertexName = poputil::templateVertex("L1DistanceGradSingleVertex", a.elementType());
    std::map<std::pair<size_t, size_t>, poplar::Tensor> bCache, gradCache;
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a1_index = j / k;
                size_t a2_index = j % k;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a1_index][a2_index]);
                graph.connect(
                    v["b"], getCachedCopy(bCache, graph, i, a2_index,
                                          b.slice({a2_index, a2_index + 1}, 1).squeeze({1}), prog));
                graph.connect(v["gradOutput"], getCachedCopy(gradCache, graph, i, a1_index,
                                                             gradOutput[a1_index], prog));
                graph.connect(v["grad"], grad[a1_index][a2_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return grad;
}

poplar::Tensor l2distance(poplar::Graph& graph,
                          const poplar::Tensor& a,
                          const poplar::Tensor& b,
                          poplar::program::Sequence& prog,
                          const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || a.dim(1) != b.dim(1)) {
        std::ostringstream msg;
        msg << "Bad arguments to l2distance, expected a.shape (M, K), b.shape (N, "
               "K), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t n = b.dim(0);
    poplar::Tensor out =
        graph.addVariable(a.elementType(), {a.dim(0), b.dim(0)}, {debugContext, "l2dist_out"});
    mapTensor2Dblocks(graph, out);
    const auto& mapping = graph.getTileMapping(out);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l2dist"});
    const auto vertexName = poputil::templateVertex("L2DistanceSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a_index = j / n;
                size_t b_index = j % n;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a_index]);
                graph.connect(v["b"], b[b_index]);
                graph.connect(v["out"], out[a_index][b_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return out;
}

poplar::Tensor l2distancegrad(poplar::Graph& graph,
                              const poplar::Tensor& a,
                              const poplar::Tensor& b,
                              const poplar::Tensor& dist,
                              const poplar::Tensor& gradOutput,
                              poplar::program::Sequence& prog,
                              const poplar::DebugContext& debugContext) {
    if (a.rank() != 2 || b.rank() != 2 || dist.rank() != 2 || gradOutput.rank() != 2 ||
        a.dim(1) != b.dim(1) || gradOutput.dim(0) != a.dim(0) || gradOutput.dim(1) != b.dim(0) ||
        dist.dim(0) != a.dim(0) || dist.dim(1) != b.dim(0)) {
        std::ostringstream msg;
        msg << "Bad arguments to l2distancegrad, expected"
            << " a.shape (M, K), b.shape (N, K), gradOutput.shape (M, N), actual"
            << " a.shape = " << a.shapeToString() << ", b.shape = " << b.shapeToString()
            << ", gradOutput.shape = " << gradOutput.shapeToString() << ".";
        throw std::invalid_argument(msg.str());
    }
    const size_t k = a.dim(1);
    poplar::Tensor grad =
        graph.addVariable(a.elementType(), a.shape(), {debugContext, "l2dist_grad"});
    mapTensor2Dblocks(graph, grad);
    const auto& mapping = graph.getTileMapping(grad);
    poplar::ComputeSet cs = graph.addComputeSet({debugContext, "l2dist_grad"});
    const auto vertexName = poputil::templateVertex("L2DistanceGradSingleVertex", a.elementType());
    for (size_t i = 0; i < mapping.size(); ++i) {
        for (const auto& interval : mapping[i]) {
            for (size_t j = interval.begin(); j != interval.end(); ++j) {
                size_t a1_index = j / k;
                size_t a2_index = j % k;
                auto v = graph.addVertex(cs, vertexName);
                graph.connect(v["a"], a[a1_index][a2_index]);
                graph.connect(v["b"], b.slice({a2_index, a2_index + 1}, 1).squeeze({1}));
                graph.connect(v["dist"], dist[a1_index]);
                graph.connect(v["gradOutput"], gradOutput[a1_index]);
                graph.connect(v["grad"], grad[a1_index][a2_index]);
                graph.setTileMapping(v, i);
                graph.setPerfEstimate(v, 0);  // placeholder
            }
        }
    }
    prog.add(poplar::program::Execute(cs));
    return grad;
}

const popart::OperatorIdentifier L1DistanceId = {"ai.graphcore.pea", "L1Distance", 1};
const popart::OperatorIdentifier L2DistanceId = {"ai.graphcore.pea", "L2Distance", 1};
const popart::OperatorIdentifier L1DistanceGradId = {"ai.graphcore.pea", "L1DistanceGrad", 1};
const popart::OperatorIdentifier L2DistanceGradId = {"ai.graphcore.pea", "L2DistanceGrad", 1};

class L1DistanceOp;
class L1DistanceGradOpx;
class L2DistanceOp;
class L2DistanceGradOpx;

class L1DistanceGradOp : public popart::Op {
   public:
    L1DistanceGradOp(const L1DistanceOp& fwdOp);

    std::unique_ptr<popart::Op> clone() const final {
        return std::make_unique<L1DistanceGradOp>(*this);
    }
    void setup() final {
        auto AInfo = inInfo(1);
        auto BInfo = inInfo(2);
        outInfo(0) = AInfo;
        outInfo(1) = BInfo;
    };

    const std::vector<popart::GradInOutMapper>& gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> inInfo = {
            {0, 0, popart::GradOpInType::GradOut},
            {1, 0, popart::GradOpInType::In},
            {2, 1, popart::GradOpInType::In}};
        return inInfo;
    }

    // The Grad Op has 1 output, which is the gradient of the only input
    const std::map<int, int>& gradOutToNonGradIn() const {
        static const std::map<int, int> outInfo = {{0, 0}, {1, 1}};
        return outInfo;
    }

    bool requiresRandomSeed() const override { return false; }

    // an estimate of how valuable sub-graph matching will be
    float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class L1DistanceOp : public popart::Op {
   public:
    std::string rootPath;

   public:
    L1DistanceOp(const popart::OperatorIdentifier& _opid,
                 const popart::Op::Settings& settings_,
                 const std::string& rootPath_)
        : popart::Op(_opid, settings_), rootPath(rootPath_) {}

    std::unique_ptr<Op> clone() const final { return std::make_unique<L1DistanceOp>(*this); }

    void setup() final {
        auto AInfo = inInfo(0);
        auto BInfo = inInfo(1);
        assert(AInfo.rank() == 2);
        assert(BInfo.rank() == 2);
        assert(AInfo.dim(1) == BInfo.dim(1));
        outInfo(0).set(AInfo.dataType(), {AInfo.dim(0), BInfo.dim(0)});
    }

    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<Op>> upops;
        upops.emplace_back(new L1DistanceGradOp(*this));
        return upops;
    }

    float getSubgraphValue() const final { return getHighSubgraphValue(); }

    bool requiresRandomSeed() const override { return false; }
};

class L2DistanceGradOp : public popart::Op {
   public:
    L2DistanceGradOp(const L2DistanceOp& fwdOp);

    std::unique_ptr<popart::Op> clone() const final {
        return std::make_unique<L2DistanceGradOp>(*this);
    }
    void setup() final {
        auto AInfo = inInfo(1);
        auto BInfo = inInfo(2);
        outInfo(0) = AInfo;
        outInfo(1) = BInfo;
    };

    const std::vector<popart::GradInOutMapper>& gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> inInfo = {
            {0, 0, popart::GradOpInType::GradOut},
            {1, 0, popart::GradOpInType::In},
            {2, 1, popart::GradOpInType::In},
            {3, 0, popart::GradOpInType::Out}};
        return inInfo;
    }

    // The Grad Op has 1 output, which is the gradient of the only input
    const std::map<int, int>& gradOutToNonGradIn() const {
        static const std::map<int, int> outInfo = {{0, 0}, {1, 1}};
        return outInfo;
    }

    bool requiresRandomSeed() const override { return false; }

    // an estimate of how valuable sub-graph matching will be
    float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class L2DistanceOp : public popart::Op {
   public:
    std::string rootPath;

   public:
    L2DistanceOp(const popart::OperatorIdentifier& _opid,
                 const popart::Op::Settings& settings_,
                 const std::string& rootPath_)
        : popart::Op(_opid, settings_), rootPath(rootPath_) {}

    std::unique_ptr<Op> clone() const final { return std::make_unique<L2DistanceOp>(*this); }

    void setup() final {
        auto AInfo = inInfo(0);
        auto BInfo = inInfo(1);
        assert(AInfo.rank() == 2);
        assert(BInfo.rank() == 2);
        assert(AInfo.dim(1) == BInfo.dim(1));
        outInfo(0).set(AInfo.dataType(), {AInfo.dim(0), BInfo.dim(0)});
    }

    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<Op>> upops;
        upops.emplace_back(new L2DistanceGradOp(*this));
        return upops;
    }

    float getSubgraphValue() const final { return getHighSubgraphValue(); }

    bool requiresRandomSeed() const override { return false; }
};

using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition l1DistanceOpDef({OpDefinition::Inputs({{"a", T}, {"b", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes({{"root_path", {"string"}}})});

static popart::OpCreator<L1DistanceOp> l1DistanceOpCreator(
    popart::OpDefinitions({{L1DistanceId, l1DistanceOpDef}}),
    [](const popart::OpCreatorInfo& info) {
        auto rootPath = info.attributes.getAttribute<popart::Attributes::String>("root_path");
        return std::make_unique<L1DistanceOp>(info.opid, info.settings, rootPath);
    },
    true);

static OpDefinition l2DistanceOpDef({OpDefinition::Inputs({{"a", T}, {"b", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes({{"root_path", {"string"}}})});

static popart::OpCreator<L2DistanceOp> l2DistanceOpCreator(
    popart::OpDefinitions({{L2DistanceId, l2DistanceOpDef}}),
    [](const popart::OpCreatorInfo& info) {
        auto rootPath = info.attributes.getAttribute<popart::Attributes::String>("root_path");
        return std::make_unique<L2DistanceOp>(info.opid, info.settings, rootPath);
    },
    true);

class L1DistanceOpx : public popart::popx::Opx {
   public:
    L1DistanceOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<L1DistanceOp>(op, {L1DistanceId});
        // add codelets to graph
        graph().addCodelets(getOp<L1DistanceOp>().rootPath + "/cpp/distance_matrix_codelet.cpp");
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<L1DistanceOp>();

        poplar::Tensor A = getInTensor(0);
        poplar::Tensor B = getInTensor(1);
        poplar::Tensor out = l1distance(graph(), A, B, prog, "l1distance");
        setOutTensor(0, out);
    }
};

class L1DistanceGradOpx : public popart::popx::Opx {
   public:
    L1DistanceGradOpx(popart::Op* op, popart::popx::Devicex* devicex)
        : popart::popx::Opx(op, devicex) {
        verifyOp<L1DistanceGradOp>(op, {L1DistanceGradId});
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<L1DistanceGradOp>();

        poplar::Tensor grad = getInTensor(0);
        poplar::Tensor A = getInTensor(1);
        poplar::Tensor B = getInTensor(2);
        auto gradA = l1distancegrad(graph(), A, B, grad, prog, "l1distanceGrad");
        auto gradB = l1distancegrad(graph(), B, A, grad.transpose(), prog, "l1distanceGrad");
        setOutTensor(0, gradA);
        setOutTensor(1, gradB);
    }
};

class L2DistanceOpx : public popart::popx::Opx {
   public:
    L2DistanceOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<L2DistanceOp>(op, {L2DistanceId});
        // add codelets to graph
        graph().addCodelets(getOp<L2DistanceOp>().rootPath + "/cpp/distance_matrix_codelet.cpp");
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<L2DistanceOp>();

        poplar::Tensor A = getInTensor(0);
        poplar::Tensor B = getInTensor(1);
        poplar::Tensor out = l2distance(graph(), A, B, prog, "l2distance");
        setOutTensor(0, out);
    }
};

class L2DistanceGradOpx : public popart::popx::Opx {
   public:
    L2DistanceGradOpx(popart::Op* op, popart::popx::Devicex* devicex)
        : popart::popx::Opx(op, devicex) {
        verifyOp<L2DistanceGradOp>(op, {L2DistanceGradId});
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<L2DistanceGradOp>();

        poplar::Tensor grad = getInTensor(0);
        poplar::Tensor A = getInTensor(1);
        poplar::Tensor B = getInTensor(2);
        poplar::Tensor dist = getInTensor(3);
        auto gradA = l2distancegrad(graph(), A, B, dist, grad, prog, "l2distanceGrad");
        auto gradB = l2distancegrad(graph(), B, A, dist.transpose(), grad.transpose(), prog,
                                    "l2distanceGrad");
        setOutTensor(0, gradA);
        setOutTensor(1, gradB);
    }
};

L1DistanceGradOp::L1DistanceGradOp(const L1DistanceOp& fwdOp)
    : popart::Op(L1DistanceGradId, fwdOp.settings) {}

L2DistanceGradOp::L2DistanceGradOp(const L2DistanceOp& fwdOp)
    : popart::Op(L2DistanceGradId, fwdOp.settings) {}

static popart::popx::OpxCreator<L1DistanceOpx> L1DistanceOpxCreator({L1DistanceId});
static popart::popx::OpxCreator<L1DistanceGradOpx> L1DistanceGradOpxCreator({L1DistanceGradId});
static popart::popx::OpxCreator<L2DistanceOpx> L2DistanceOpxCreator({L2DistanceId});
static popart::popx::OpxCreator<L2DistanceGradOpx> L2DistanceGradOpxCreator({L2DistanceGradId});

}  // namespace
