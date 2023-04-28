// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <memory>
#include <vector>

#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <popsparse/codelets.hpp>

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
popsparse::static_::PlanningCache* sparsePlanningCache() {
    static popsparse::static_::PlanningCache cache;
    return &cache;
}

template <class T>
popsparse::CSRMatrix<T> cooToCsr(const popsparse::COOMatrix<T>& matrix) {
    auto blockRowSize = matrix.getBlockDimensions()[0];
    auto blockElements = matrix.getBlockSize();
    std::vector<T> nzValues(matrix.nzValues.size());
    std::vector<size_t> columnIndices(matrix.columnIndices.size());
    std::vector<size_t> rowIndices(matrix.numRows / blockRowSize + 1);

    std::vector<size_t> ordering(matrix.columnIndices.size());
    std::iota(ordering.begin(), ordering.end(), 0u);
    std::sort(ordering.begin(), ordering.end(), [&matrix](size_t a, size_t b) {
        if (matrix.rowIndices[a] != matrix.rowIndices[b]) {
            return matrix.rowIndices[a] < matrix.rowIndices[b];
        }
        return matrix.columnIndices[a] < matrix.columnIndices[b];
    });

    auto row = 1u;
    for (auto i = 0u; i < ordering.size(); ++i) {
        columnIndices[i] = matrix.columnIndices[ordering[i]];
        std::copy(matrix.nzValues.begin() + ordering[i] * blockElements,
                  matrix.nzValues.begin() + (ordering[i] + 1) * blockElements,
                  nzValues.begin() + i * blockElements);
        while (blockRowSize * (row - 1) < matrix.rowIndices[ordering[i]]) {
            rowIndices[row++] = i * blockElements;
        }
    }
    while (row < rowIndices.size()) {
        rowIndices[row++] = nzValues.size();
    }

    return popsparse::CSRMatrix<T>(matrix.numRows, matrix.numColumns, nzValues, columnIndices,
                                   rowIndices, matrix.getBlockDimensions());
}

poplar::Tensor constSparseAndDenseMatMul(poplar::Graph& graph,
                                         const popsparse::CSRMatrix<float>& sparse,
                                         const poplar::Tensor& dense,
                                         poplar::program::Sequence& prog,
                                         bool sparseLhs,
                                         const poplar::DebugContext& debugContext) {
    poplar::OptionFlags options;

    popsparse::static_::MatMulParams params;
    popsparse::static_::SparseTensor dummySparseTensor;

    if (sparseLhs) {
        params = popsparse::static_::MatMulParams::createForSparseDense(
            /*groups*/ 1, sparse.numRows, sparse.numColumns, dense.dim(1));

        dummySparseTensor = popsparse::static_::createSparseDenseMatMulLHS(
            graph, dense.elementType(), params, sparse, {debugContext, "lhs"}, options,
            sparsePlanningCache());

    } else {  // sparseRhs
        params = popsparse::static_::MatMulParams::createForDenseSparse(
            /*groups*/ 1, dense.dim(0), sparse.numRows, sparse.numColumns);

        dummySparseTensor = popsparse::static_::createDenseSparseMatMulRHS(
            graph, dense.elementType(), params, sparse, {debugContext, "rhs"}, options,
            sparsePlanningCache());
    }

    auto nzValues =
        popsparse::static_::Partitioner<float>(params, dense.elementType(), graph.getTarget(),
                                               options, sparsePlanningCache())
            .createSparsityDataImpl(sparse)
            .nzValues;
    auto nzValuesTensor =
        graph.addConstant<float>(dense.elementType(), dummySparseTensor.getNzValuesTensor().shape(),
                                 nzValues, {debugContext, sparseLhs ? "lhs" : "rhs"});
    graph.setTileMapping(nzValuesTensor,
                         graph.getTileMapping(dummySparseTensor.getNzValuesTensor()));
    auto sparseTensor =
        popsparse::static_::SparseTensor(nzValuesTensor, dummySparseTensor.getOpMetaData());

    if (sparseLhs) {
        return popsparse::static_::sparseDenseMatMul(
                   graph, sparseTensor, dense.expand({0}), prog, /*transposeLHS*/ false,
                   /*transposeRHS*/ false, {debugContext, "output"}, options, sparsePlanningCache())
            .squeeze({0});

    } else {  // sparseRhs
        return popsparse::static_::denseSparseMatMul(
                   graph, dense.expand({0}), sparseTensor, prog, /*transposeLHS*/ false,
                   /*transposeRHS*/ false, {debugContext, "output"}, options, sparsePlanningCache())
            .squeeze({0});
    }
}

struct CustomOp : popart::Op {
    static const popart::OperatorIdentifier ID;
    popsparse::CSRMatrix<float> matrix;
    std::string mode;

    CustomOp(const popart::OperatorIdentifier& opid_,
             const popart::Op::Settings& settings_,
             popsparse::CSRMatrix<float>&& matrix_,
             const std::string& mode_)
        : popart::Op(opid_, settings_), matrix(matrix_), mode(mode_) {}

    std::unique_ptr<Op> clone() const final { return std::make_unique<CustomOp>(*this); }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    void setup() {
        auto& input = inInfo(0);
        outInfo(0) = {
            input.dataType(),
            mode == "sparse_dense"
                ? std::vector<int64_t>{static_cast<int>(matrix.numRows), input.dim(1)}
                : std::vector<int64_t>{input.dim(0), static_cast<int>(matrix.numColumns)}};
    }
    void appendAttributes(popart::OpSerialiserBase& os) const final {
        popart::Op::appendAttributes(os);
        appendLocalAttributes(os);
    }
    void appendOutlineAttributes(popart::OpSerialiserBase& os) const final {
        popart::Op::appendOutlineAttributes(os);
        appendLocalAttributes(os);
    }

   private:
    template <class T>
    static std::string vectorToString(const std::vector<T>& v) {
        std::ostringstream str;
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(str, " "));
        return str.str();
    }
    void appendLocalAttributes(popart::OpSerialiserBase& os) const {
        os.appendAttribute("mode", mode);
        os.appendAttribute("numRows", matrix.numRows);
        os.appendAttribute("numColumns", matrix.numColumns);
        os.appendAttribute("blockSizeRows", matrix.getBlockDimensions()[0]);
        os.appendAttribute("blockSizeColumns", matrix.getBlockDimensions()[1]);
        os.appendAttribute("rowIndices", vectorToString(matrix.rowIndices));
        os.appendAttribute("columnIndices", vectorToString(matrix.columnIndices));
        os.appendAttribute("nzValues", vectorToString(matrix.nzValues));
    }
};

struct CustomOpx : popart::popx::Opx {
    CustomOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<CustomOp>(op, CustomOp::ID);
    }
    void grow(poplar::program::Sequence& prog) const final {
        popsparse::addCodelets(graph());  // Note: this might not belong here
        auto& op = getOp<CustomOp>();
        auto input = get(inId(0));
        auto output = constSparseAndDenseMatMul(graph(), op.matrix, input, prog,
                                                /*sparseLhs*/ op.mode == "sparse_dense",
                                                debugContext("StaticSparseMatmul"));
        insert(outId(0), output);
    }
};

// We cannot use "ai.graphcore.pea", since shape inference tries to call
// `Ir::getDefaultOpsetVersion` which cannot be extended to custom domains
const popart::OperatorIdentifier CustomOp::ID = {"ai.graphcore", "StaticSparseMatmul", 1};
popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16, popart::DataType::FLOAT};
popart::OpCreator<CustomOp> opCreator(
    {{CustomOp::ID,
      {popart::OpDefinition::Inputs({{"input", T}}), popart::OpDefinition::Outputs({{"output", T}}),
       popart::OpDefinition::Attributes({{"mode", {"string"}},
                                         {"n_rows", {"int"}},
                                         {"n_cols", {"int"}},
                                         {"block_size", {"int"}},
                                         {"rows", {"*"}},
                                         {"cols", {"*"}},
                                         {"values", {"*"}}})}}},
    [](const popart::OpCreatorInfo& info) {
        auto mode = info.attributes.getAttribute<popart::Attributes::String>("mode");
        auto nRows = info.attributes.getAttribute<popart::Attributes::Int>("n_rows");
        auto nCols = info.attributes.getAttribute<popart::Attributes::Int>("n_cols");
        unsigned blockSize = info.attributes.getAttribute<popart::Attributes::Int>("block_size");
        auto rows = info.attributes.getAttribute<popart::Attributes::Ints>("rows");
        auto cols = info.attributes.getAttribute<popart::Attributes::Ints>("cols");
        auto values = info.attributes.getAttribute<popart::Attributes::Floats>("values");
        auto matrix =
            popsparse::COOMatrix<float>(nRows, nCols, values, {cols.begin(), cols.end()},
                                        {rows.begin(), rows.end()}, {blockSize, blockSize});
        return std::make_unique<CustomOp>(info.opid, info.settings, cooToCsr(matrix), mode);
    },
    true);
popart::popx::OpxCreator<CustomOpx> opxCreator(CustomOp::ID);
}  // namespace
