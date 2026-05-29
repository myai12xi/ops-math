/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/*!
 * \file reduce_nansum_tiling.cpp
 * \brief ReduceNansum Tiling 实现（通用，支持 arch32/arch35）
 * 迭代三：支持全部 4 个 TilingKey 分支（AR-Fullload, AR-ColSplit, ARA-Fullload, ARA-RowSplit），
 * 多核并行，任意 axis 归约，全 dtype（fp32/fp16/bf16），边界处理。
 *
 * Tiling 通过 platform API (GetCoreNumAiv, GetCoreMemSize) 动态获取平台参数，
 * 因此同一份代码可适配 arch32 (Ascend910B) 和 arch35 (Ascend950) 等架构。
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/common/reduce_nansum_tiling_data.h"
#include "../../op_kernel/common/reduce_nansum_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t MIN_TMP_BUF_SIZE = 4096U;
constexpr uint32_t VECTOR_REG_WIDTH = 256U;
constexpr uint32_t UB_RESERVE = 8192U;
constexpr uint32_t SCH_AR_FULLLOAD = 0;
constexpr uint32_t SCH_AR_COLSPLIT = 1;
constexpr uint32_t SCH_ARA_FULLLOAD = 2;
constexpr uint32_t SCH_ARA_ROWSPLIT = 3;
constexpr int64_t MAX_DIM = 8;

// AR 全载阈值上限（保守值）
constexpr int64_t AR_FULLLOAD_THRESHOLD_MAX = 12000;

// ARA 基础 tile 宽度：fp32 对齐到 256 字节 = 64 个 float
constexpr int64_t A0_TILE_BASE = 64;

// ARA R_max 上限（repeatTimes <= 255）
constexpr int64_t ARA_R_MAX_UPPER = 255;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) return g_vec_1_shape;
    return inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline int64_t AlignTo32Bytes(int64_t count, int64_t typeSize)
{
    if (typeSize == 0) {
        return count;
    }
    int64_t bytes = count * typeSize;
    int64_t alignedBytes = ((bytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    return alignedBytes / typeSize;
}

// Compare API 要求 count 个元素所占空间 256 字节对齐
static inline int64_t AlignTo256Bytes(int64_t count, int64_t typeSize)
{
    if (typeSize == 0) {
        return count;
    }
    int64_t bytes = count * typeSize;
    int64_t alignedBytes = ((bytes + VECTOR_REG_WIDTH - 1) / VECTOR_REG_WIDTH) * VECTOR_REG_WIDTH;
    return alignedBytes / typeSize;
}

static inline int64_t ComputeReduceTmpBufSize(int64_t totalElements)
{
    int64_t perRepeat = VECTOR_REG_WIDTH / static_cast<int64_t>(sizeof(float));
    int64_t perBlock = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
    int64_t repeats = CeilDiv(totalElements, perRepeat);
    int64_t tmpBufSize = CeilDiv(repeats, perBlock) * perBlock * static_cast<int64_t>(sizeof(float));
    if (tmpBufSize < static_cast<int64_t>(MIN_TMP_BUF_SIZE)) tmpBufSize = static_cast<int64_t>(MIN_TMP_BUF_SIZE);
    return tmpBufSize;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// axes 参数解析：从 Tensor 输入读取，规范化负索引、排序去重
// ============================================================================
static void ParseDimAttr(gert::TilingContext* context, const gert::Shape& inputShapeX,
                         int64_t dimArr[], int64_t& dimCount, bool& isFullReduce)
{
    int64_t rank = static_cast<int64_t>(inputShapeX.GetDimNum());
    dimCount = 0;
    isFullReduce = true;

    // 获取 axes 输入（index=1）
    auto axesTensor = context->GetInputTensor(1);
    if (axesTensor == nullptr) {
        return;
    }

    auto axesDesc = context->GetInputDesc(1);
    if (axesDesc == nullptr) {
        return;
    }

    int64_t listSize = axesTensor->GetShapeSize();
    if (listSize <= 0) {
        return;
    }

    auto axesDtype = axesDesc->GetDataType();

    // 规范化负索引
    if (axesDtype == ge::DT_INT64) {
        const int64_t* data = axesTensor->GetData<int64_t>();
        if (data == nullptr) return;
        for (int64_t i = 0; i < listSize && dimCount < MAX_DIM; i++) {
            int64_t d = data[i];
            if (d < 0) d += rank;
            if (d >= 0 && d < rank) {
                dimArr[dimCount++] = d;
            }
        }
    } else {
        const int32_t* data = axesTensor->GetData<int32_t>();
        if (data == nullptr) return;
        for (int64_t i = 0; i < listSize && dimCount < MAX_DIM; i++) {
            int64_t d = static_cast<int64_t>(data[i]);
            if (d < 0) d += rank;
            if (d >= 0 && d < rank) {
                dimArr[dimCount++] = d;
            }
        }
    }

    if (dimCount == 0) return;

    // 排序
    for (int64_t i = 0; i < dimCount - 1; i++) {
        for (int64_t j = i + 1; j < dimCount; j++) {
            if (dimArr[i] > dimArr[j]) {
                int64_t tmp = dimArr[i];
                dimArr[i] = dimArr[j];
                dimArr[j] = tmp;
            }
        }
    }

    // 去重
    int64_t uniqueCount = 1;
    for (int64_t i = 1; i < dimCount; i++) {
        if (dimArr[i] != dimArr[uniqueCount - 1]) {
            dimArr[uniqueCount++] = dimArr[i];
        }
    }
    dimCount = uniqueCount;

    // 检查是否等价于全量归约
    if (dimCount == rank) {
        isFullReduce = true;
        return;
    }

    isFullReduce = false;
}

// ============================================================================
// 3D 抽象：将任意 shape + dim 映射为 (A1, R, A0)
// 对于非连续归约轴（如 dim=[0,2]），输出 strided copy 参数
// ============================================================================
struct StridedCopyInfo {
    bool isStrided = false;        // 是否为非连续多轴归约
    int64_t blockCount = 0;        // 每行由多少个块组成
    int64_t innerBlockSize = 0;    // 每个块的大小（元素数）
    int64_t gapBetweenBlocks = 0;  // 块之间的 GM 间距（元素数）
    int64_t outputStride = 0;      // 输出元素之间的步长（元素数）- 仅单非归约维度
    // 非归约维度信息（用于正确计算每行的 GM 偏移）
    int64_t nonReduceDimCount = 0;
    int64_t nonReduceDimSizes[MAX_DIM] = {0};
    int64_t nonReduceGmStrides[MAX_DIM] = {0};
    // 归约维度信息（用于逐元素计算 GM 偏移）
    int64_t reduceDimCount = 0;
    int64_t reduceDimSizes[MAX_DIM] = {0};
    int64_t reduceGmStrides[MAX_DIM] = {0};
};

static void Compute3DAbstraction(const gert::Shape& inputShapeX,
                                 const int64_t dimArr[], int64_t dimCount, bool isFullReduce,
                                 int64_t& a1Count, int64_t& rCount, int64_t& a0Count,
                                 StridedCopyInfo& stridedInfo)
{
    int64_t rank = static_cast<int64_t>(inputShapeX.GetDimNum());
    a1Count = 1;
    rCount = 1;
    a0Count = 1;
    stridedInfo = StridedCopyInfo();

    if (isFullReduce) {
        // dim=None: A1=1, R=totalElements, A0=1
        for (int64_t i = 0; i < rank; i++) {
            rCount *= inputShapeX.GetDim(i);
        }
        return;
    }

    // 检查归约轴是否连续
    bool isContiguous = true;
    for (int64_t i = 1; i < dimCount; i++) {
        if (dimArr[i] != dimArr[i - 1] + 1) {
            isContiguous = false;
            break;
        }
    }

    if (isContiguous) {
        // 连续轴情况：直接映射为 (A1, R, A0)
        int64_t minDim = dimArr[0];
        int64_t maxDim = dimArr[dimCount - 1];

        for (int64_t i = 0; i < minDim; i++) {
            a1Count *= inputShapeX.GetDim(i);
        }
        for (int64_t i = minDim; i <= maxDim; i++) {
            rCount *= inputShapeX.GetDim(i);
        }
        for (int64_t i = maxDim + 1; i < rank; i++) {
            a0Count *= inputShapeX.GetDim(i);
        }
        return;
    }

    // =========== 非连续轴情况 ===========
    // 标记哪些维度是归约维度
    bool isReduceDim[MAX_DIM] = {false};
    for (int64_t i = 0; i < dimCount; i++) {
        isReduceDim[dimArr[i]] = true;
    }

    int64_t lastReduceDim = dimArr[dimCount - 1];
    int64_t firstReduceDim = dimArr[0];

    // 计算 inputStrides（所有维度的 GM 步长）
    int64_t inputStrides[MAX_DIM];
    inputStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        inputStrides[i] = inputStrides[i + 1] * inputShapeX.GetDim(i + 1);
    }

    // 计算 R: 归约维度的乘积
    rCount = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (isReduceDim[i]) {
            rCount *= inputShapeX.GetDim(i);
        }
    }

    // 检查最后一个归约轴之后是否有非归约维度（trailing non-reduce dims）
    // 如果有，需要将它们作为 A0 维度，使用 ARA 模板
    int64_t trailingA0 = 1;
    for (int64_t i = lastReduceDim + 1; i < rank; i++) {
        trailingA0 *= inputShapeX.GetDim(i);
    }

    if (trailingA0 > 1) {
        // ===== 子情况B：有 trailing non-reduce dims → ARA 模板 =====
        // 例如 dim=[0,2] on shape(4,6,8,10):
        //   A0 = 10 (trailing non-reduce)
        //   A1 = 6 (其余 non-reduce dims)
        //   R = 4*8 = 32
        //   DataCopyPad: 用 strided copy 将数据搬入 UB 连续布局 (R, A0)
        a0Count = trailingA0;
        a1Count = 1;
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduceDim[i] && i <= lastReduceDim) {
                a1Count *= inputShapeX.GetDim(i);
            }
        }

        // 收集非归约维度信息（仅首个归约轴之前及归约轴之间的，不含 trailing）
        int64_t nrCount = 0;
        for (int64_t i = 0; i <= lastReduceDim; i++) {
            if (!isReduceDim[i]) {
                stridedInfo.nonReduceDimSizes[nrCount] = inputShapeX.GetDim(i);
                stridedInfo.nonReduceGmStrides[nrCount] = inputStrides[i];
                nrCount++;
            }
        }
        stridedInfo.nonReduceDimCount = nrCount;

        // 收集归约维度信息
        int64_t rdCount = 0;
        for (int64_t i = 0; i < rank; i++) {
            if (isReduceDim[i]) {
                stridedInfo.reduceDimSizes[rdCount] = inputShapeX.GetDim(i);
                stridedInfo.reduceGmStrides[rdCount] = inputStrides[i];
                rdCount++;
            }
        }
        stridedInfo.reduceDimCount = rdCount;

        // 计算 ARA strided copy 参数:
        // 从最后一个归约轴到最后一个维度，连续归约轴及其后的所有维度构成一个"内块"
        // innerBlockSize = product(dims from lastReduceDim to rank-1) = lastReduceDim's size * trailingA0
        // blockCount = R / (lastReduceDim's contiguous reduce group size)
        // 计算最后一组连续归约轴的大小（从 lastReduceDim 向前直到遇到非归约轴或首个轴）
        int64_t innerReduceGroupSize = 1;
        for (int64_t i = lastReduceDim; i >= 0; i--) {
            if (isReduceDim[i]) {
                innerReduceGroupSize *= inputShapeX.GetDim(i);
            } else {
                break;
            }
        }
        int64_t innerBlockSize = innerReduceGroupSize * trailingA0;
        int64_t blockCount = rCount / innerReduceGroupSize;

        // 计算外层归约步长
        // 外层归约轴的 GM 步长 = inputStrides[firstReduceDim]
        // 每个外层块在 GM 中的间距 = inputStrides[firstReduceDim]
        // srcStride (block尾到下一block头) = inputStrides[firstReduceDim] - innerBlockSize
        int64_t outerStride = inputStrides[firstReduceDim];
        int64_t gmStrideElements = outerStride - innerBlockSize;

        stridedInfo.isStrided = true;
        stridedInfo.blockCount = blockCount;
        stridedInfo.innerBlockSize = innerBlockSize;
        stridedInfo.gapBetweenBlocks = gmStrideElements;
        stridedInfo.outputStride = 0;  // ARA 模板不使用此参数

    } else {
        // ===== 子情况A：无 trailing non-reduce dims → AR 模板 =====
        // 例如 dim=[0,2] on shape(3,4,5):
        //   A0 = 1
        //   A1 = 4 (non-reduce dim 1)
        //   R = 3*5 = 15
        //   innerBlockSize = 5 (dim 2 是最后一维，连续)
        a0Count = 1;
        a1Count = 1;
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduceDim[i]) {
                a1Count *= inputShapeX.GetDim(i);
            }
        }

        // 计算最后一组连续归约轴的内层块大小
        int64_t innerBlockSize = 1;
        for (int64_t i = lastReduceDim; i < rank; i++) {
            if (isReduceDim[i]) {
                innerBlockSize *= inputShapeX.GetDim(i);
            } else {
                break;
            }
        }

        // blockCount = R / innerBlockSize
        int64_t blockCount = rCount / innerBlockSize;

        // 计算外层步长
        int64_t tailProduct = 1;
        for (int64_t i = firstReduceDim + 1; i < rank; i++) {
            tailProduct *= inputShapeX.GetDim(i);
        }
        int64_t gmStrideElements = tailProduct - innerBlockSize;

        // 收集非归约维度信息
        int64_t nrCount = 0;
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduceDim[i]) {
                stridedInfo.nonReduceDimSizes[nrCount] = inputShapeX.GetDim(i);
                stridedInfo.nonReduceGmStrides[nrCount] = inputStrides[i];
                nrCount++;
            }
        }
        stridedInfo.nonReduceDimCount = nrCount;

        // 收集归约维度信息
        int64_t rdCount = 0;
        for (int64_t i = 0; i < rank; i++) {
            if (isReduceDim[i]) {
                stridedInfo.reduceDimSizes[rdCount] = inputShapeX.GetDim(i);
                stridedInfo.reduceGmStrides[rdCount] = inputStrides[i];
                rdCount++;
            }
        }
        stridedInfo.reduceDimCount = rdCount;

        // outputStride: 保留兼容（仅单非归约维度时使用）
        int64_t inputRowStride = 0;
        for (int64_t i = firstReduceDim; i <= lastReduceDim; i++) {
            if (!isReduceDim[i]) {
                inputRowStride = 1;
                for (int64_t j = i + 1; j < rank; j++) {
                    inputRowStride *= inputShapeX.GetDim(j);
                }
                break;
            }
        }

        stridedInfo.isStrided = true;
        stridedInfo.blockCount = blockCount;
        stridedInfo.innerBlockSize = innerBlockSize;
        stridedInfo.gapBetweenBlocks = gmStrideElements;
        stridedInfo.outputStride = inputRowStride;
    }
}

// ============================================================================
// AR 全载阈值计算（dtype-aware）
// ============================================================================
static int64_t ComputeArFullloadThreshold(int64_t ubSize, [[maybe_unused]] int64_t typeSize, bool isMixedPrecision)
{
    // overhead = outQueueY(2 * 32) + tmpBuf(4096) = 4160
    int64_t overhead = 4160;
    int64_t perElement;

    if (!isMixedPrecision) {
        // fp32 直接计算路径:
        // inQueueX(2 * N * 4) + maskBuf(N/8, 对齐忽略) + zeroBuf(N * 4) + cleanBuf(N * 4)
        // 简化: N * (8 + 0.125 + 4 + 4) ≈ N * 14
        perElement = 14;
    } else {
        // fp16/bf16 混合精度路径:
        // inQueueX(2 * N * 2) + maskBuf(N/8) + zeroBuf(N * 4) + cleanBuf(N * 4) + castBuf(N * 4)
        // 简化: N * (4 + 0.125 + 4 + 4 + 4) ≈ N * 16
        perElement = 16;
    }

    int64_t threshold = (ubSize - overhead) / perElement;
    if (threshold > AR_FULLLOAD_THRESHOLD_MAX) {
        threshold = AR_FULLLOAD_THRESHOLD_MAX;
    }
    if (threshold < 1) threshold = 1;
    return threshold;
}

// ============================================================================
// ARA R_max 计算（dtype-aware）
// ============================================================================
static int64_t ComputeAraRMax(int64_t ubSize, int64_t a0TileBase, bool isMixedPrecision)
{
    // overhead = a0TileBase * 12 * sizeof(float) + 4096
    int64_t overhead = a0TileBase * 12 * static_cast<int64_t>(sizeof(float)) + 4096;
    int64_t bytesPerRowPerCol;

    if (!isMixedPrecision) {
        // fp32: inQueueX(4) + maskBuf(1/8) + cleanBuf(4) ≈ 10 * sizeof(float) per col per row
        bytesPerRowPerCol = 10 * static_cast<int64_t>(sizeof(float));
    } else {
        // fp16/bf16: inQueueX(2) + maskBuf(1/8) + cleanBuf(4) + castBuf(4) ≈ 10.125 per col per row
        // Use 12 bytes per col per row for safety (includes castBuf overhead)
        bytesPerRowPerCol = 12 * static_cast<int64_t>(sizeof(float));
    }

    int64_t denominator = a0TileBase * bytesPerRowPerCol;
    int64_t rMax = (ubSize - overhead) / denominator;

    if (rMax > ARA_R_MAX_UPPER) rMax = ARA_R_MAX_UPPER;
    if (rMax < 1) rMax = 1;
    return rMax;
}

// ============================================================================
// ARA tileA0Len 计算（三约束取最小，dtype-aware）
// ============================================================================
static int64_t ComputeTileA0Len(int64_t ubSize, int64_t A0, int64_t R, [[maybe_unused]] int64_t A1,
                                int64_t a0TileBase, [[maybe_unused]] int64_t coreNum, bool isMixedPrecision)
{
    int64_t overhead = a0TileBase * 12 * static_cast<int64_t>(sizeof(float)) + 4096;
    int64_t bytesPerRowPerCol = isMixedPrecision ?
        12 * static_cast<int64_t>(sizeof(float)) :
        10 * static_cast<int64_t>(sizeof(float));
    int64_t ubPerTileBase = R * a0TileBase * bytesPerRowPerCol;

    // 约束1: UB 容量限制
    int64_t factorMax = 1;
    if (ubPerTileBase > 0) {
        factorMax = (ubSize - overhead) / ubPerTileBase;
    }
    if (factorMax < 1) factorMax = 1;

    // 约束2: A0 维度限制
    int64_t a0FactorMax = CeilDiv(A0, a0TileBase);

    // 约束3: 多核负载均衡限制（保留注释，变量已移除）
    // 不需要此约束限制 a0Inner，保留较大 tile 可减少迭代次数

    // 取最小值
    int64_t a0Inner = factorMax;
    if (a0Inner > a0FactorMax) a0Inner = a0FactorMax;
    if (a0Inner < 1) a0Inner = 1;

    int64_t tileA0Len = a0Inner * a0TileBase;
    if (tileA0Len > A0) tileA0Len = A0;
    return tileA0Len;
}

// ============================================================================
// AR-ColSplit chunk 参数计算
// ============================================================================
static void ComputeArColSplitParams(int64_t rCount, int64_t fullLoadThreshold,
                                    int64_t& rChunkSize, int64_t& numChunks, int64_t& lastChunkSize)
{
    // chunk size 取全载阈值（保证每个 chunk 能放进 UB）
    rChunkSize = fullLoadThreshold;
    numChunks = CeilDiv(rCount, rChunkSize);
    lastChunkSize = rCount - (numChunks - 1) * rChunkSize;
    if (lastChunkSize <= 0) lastChunkSize = rChunkSize;
}

// ============================================================================
// ARA-RowSplit chunk 参数计算
// ============================================================================
static void ComputeAraRowSplitParams(int64_t rCount, int64_t rMax,
                                     int64_t& rChunkSize, int64_t& numChunks, int64_t& lastChunkSize)
{
    rChunkSize = rMax;
    numChunks = CeilDiv(rCount, rChunkSize);
    lastChunkSize = rCount - (numChunks - 1) * rChunkSize;
    if (lastChunkSize <= 0) lastChunkSize = rChunkSize;
}

// ============================================================================
// Main Tiling Function
// ============================================================================
static ge::graphStatus ReduceNansumTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSizeU64;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSizeU64, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    int64_t ubSize = static_cast<int64_t>(ubSizeU64);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    int64_t totalElements = inputShapeX.GetShapeSize();

    // 空 tensor 处理：当输入总元素数为 0 时（某维度为 0），
    // nansum 的语义是输出 0。设置最小 tiling 参数，kernel 中 rowCount=0 不执行计算。
    // 但框架通常不会传入空 tensor，这里仍返回失败以保持兼容。
    OP_CHECK_IF(totalElements <= 0, OP_LOGE(context, "totalElements <= 0"), return ge::GRAPH_FAILED);

    // ================================================================
    // Step 1: 解析 dim 属性
    // ================================================================
    int64_t rank = static_cast<int64_t>(inputShapeX.GetDimNum());
    bool isFullReduce = true;
    int64_t dimArr[MAX_DIM];
    int64_t dimCount = 0;
    ParseDimAttr(context, inputShapeX, dimArr, dimCount, isFullReduce);

    OP_LOGI(context, "ReduceNansum Tiling: rank=%ld dimCount=%ld isFullReduce=%d totalElements=%ld",
            rank, dimCount, (int)isFullReduce, totalElements);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ReduceNansumTilingData* tiling = context->GetTilingData<ReduceNansumTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReduceNansumTilingData), 0, sizeof(ReduceNansumTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // ================================================================
    // Step 2: 3D 抽象
    // ================================================================
    int64_t a1Count = 1, rCount = 1, a0Count = 1;
    StridedCopyInfo stridedInfo;
    Compute3DAbstraction(inputShapeX, dimArr, dimCount, isFullReduce,
                         a1Count, rCount, a0Count, stridedInfo);

    OP_LOGI(context, "ReduceNansum 3D: A1=%ld R=%ld A0=%ld strided=%d",
            a1Count, rCount, a0Count, (int)stridedInfo.isStrided);

    // 迭代三：dtype-aware tiling
    int64_t typeSize;
    bool isMixedPrecision;
    if (dataType == ge::DT_FLOAT) {
        typeSize = 4;
        isMixedPrecision = false;
    } else {
        // fp16, bf16 - 2 bytes, mixed precision (compute in fp32)
        typeSize = 2;
        isMixedPrecision = true;
    }

    // ================================================================
    // Step 3: 分支选择 + Tiling 参数计算
    // ================================================================
    uint32_t schMode = SCH_AR_FULLLOAD;  // 默认
    int64_t rLengthAlign = 0;
    int64_t alignedCols = 0;
    int64_t tileA0Len = 0;
    int64_t a0Outer = 0;
    int64_t rChunkSize = 0;
    int64_t numChunks = 0;
    int64_t lastChunkSize = 0;
    int64_t tmpBufSize = 0;

    // 多核切分参数
    int64_t usedCoreNum = 1;
    int64_t tilesPerCore = 1;
    int64_t tailCoreTiles = 1;

    if (a0Count == 1) {
        // ========== AR 模板（A0=1）==========
        int64_t fullLoadThreshold = ComputeArFullloadThreshold(ubSize, typeSize, isMixedPrecision);
        OP_LOGI(context, "AR branch: fullLoadThreshold=%ld rCount=%ld dtype=%d mixed=%d",
                fullLoadThreshold, rCount, (int)dataType, (int)isMixedPrecision);

        // Compare API 要求元素数 * sizeof(type) 是 256 字节对齐
        // 对于混合精度：需要同时满足 fp32 对齐（用于 ReduceSum/Select）和原始 dtype 对齐（用于 Compare）
        // 取两者中更大的对齐值
        int64_t computeTypeSize = static_cast<int64_t>(sizeof(float));
        rLengthAlign = AlignTo256Bytes(rCount, computeTypeSize);
        if (isMixedPrecision) {
            int64_t rLengthAlignInput = AlignTo256Bytes(rCount, typeSize);
            if (rLengthAlignInput > rLengthAlign) rLengthAlign = rLengthAlignInput;
        }
        int64_t minVecElements = static_cast<int64_t>(VECTOR_REG_WIDTH) / typeSize;  // 按输入类型计算最小元素数
        int64_t minVecElementsFp32 = static_cast<int64_t>(VECTOR_REG_WIDTH) / computeTypeSize;
        if (rLengthAlign < minVecElements) rLengthAlign = minVecElements;
        if (rLengthAlign < minVecElementsFp32) rLengthAlign = minVecElementsFp32;

        if (rCount <= fullLoadThreshold) {
            // AR-全载
            schMode = SCH_AR_FULLLOAD;
            tmpBufSize = ComputeReduceTmpBufSize(rCount);
        } else {
            // AR-ColSplit
            schMode = SCH_AR_COLSPLIT;
            ComputeArColSplitParams(rCount, fullLoadThreshold, rChunkSize, numChunks, lastChunkSize);

            // rLengthAlign 按 chunk size 256 字节对齐（Compare API 要求）
            rLengthAlign = AlignTo256Bytes(rChunkSize, computeTypeSize);
            if (isMixedPrecision) {
                int64_t rLengthAlignInput = AlignTo256Bytes(rChunkSize, typeSize);
                if (rLengthAlignInput > rLengthAlign) rLengthAlign = rLengthAlignInput;
            }
            if (rLengthAlign < minVecElements) rLengthAlign = minVecElements;
            if (rLengthAlign < minVecElementsFp32) rLengthAlign = minVecElementsFp32;

            tmpBufSize = ComputeReduceTmpBufSize(rChunkSize);
        }

        // AR 多核切分：按 A1 行切分
        int64_t rowsPerCore = CeilDiv(a1Count, coreNum);
        usedCoreNum = CeilDiv(a1Count, rowsPerCore);
        tilesPerCore = rowsPerCore;
        tailCoreTiles = a1Count - (usedCoreNum - 1) * rowsPerCore;

        // AtomicAdd 优化：当 A1=1 且 AR-ColSplit（大R）时，按 R 维度多核切分
        if (a1Count == 1 && schMode == SCH_AR_COLSPLIT && coreNum > 1) {
            // 每核至少处理 fullLoadThreshold 个 R 元素（即1个chunk）
            int64_t maxCores = CeilDiv(rCount, fullLoadThreshold);
            usedCoreNum = coreNum;
            if (usedCoreNum > maxCores) usedCoreNum = maxCores;
            if (usedCoreNum < 1) usedCoreNum = 1;
            int64_t rPerCoreVal = CeilDiv(rCount, usedCoreNum);
            // 对齐到 fullLoadThreshold（确保每核的 R 是 chunk 的整数倍）
            rPerCoreVal = CeilDiv(rPerCoreVal, fullLoadThreshold) * fullLoadThreshold;
            // 重新计算实际使用核数
            usedCoreNum = CeilDiv(rCount, rPerCoreVal);
            if (usedCoreNum < 1) usedCoreNum = 1;
            // 多核切分不再按 A1 行切，而是每核处理1行（A1=1）
            tilesPerCore = 1;
            tailCoreTiles = 1;
            // 记录 AtomicAdd 参数
            tiling->useAtomicAdd = 1;
            tiling->totalRForSplit = rCount;
            tiling->rPerCore = rPerCoreVal;
        }

    } else {
        // ========== ARA 模板（A0>1）==========
        int64_t a0TileBase = A0_TILE_BASE;
        int64_t computeTypeSize = static_cast<int64_t>(sizeof(float));

        // 计算 tileA0Len（初步估算）
        tileA0Len = ComputeTileA0Len(ubSize, a0Count, rCount, a1Count, a0TileBase, coreNum, isMixedPrecision);

        // alignedCols: tileA0Len 对齐到 256 字节（按 fp32 计算精度对齐）
        // Compare/Select/ReduceSum 等向量 API 要求 count*sizeof(float) 是 256 字节的倍数，
        // 因此 alignedCols 必须是 64 的倍数（64 * sizeof(float) = 256）。
        // 同时需满足输入类型 256 字节对齐（用于混合精度 Compare）
        alignedCols = AlignTo256Bytes(tileA0Len, computeTypeSize);
        if (isMixedPrecision) {
            int64_t alignedColsInput = AlignTo256Bytes(tileA0Len, typeSize);
            if (alignedColsInput > alignedCols) alignedCols = alignedColsInput;
        }
        int64_t minVecElements = static_cast<int64_t>(VECTOR_REG_WIDTH) / typeSize;
        int64_t minVecElementsFp32 = static_cast<int64_t>(VECTOR_REG_WIDTH) / computeTypeSize;
        if (alignedCols < minVecElements) alignedCols = minVecElements;
        if (alignedCols < minVecElementsFp32) alignedCols = minVecElementsFp32;

        // 使用实际 alignedCols 重新计算 rMax（确保 UB 容量正确估算）
        int64_t rMax = ComputeAraRMax(ubSize, alignedCols, isMixedPrecision);

        // a0Outer: A0 方向的 tile 数
        a0Outer = CeilDiv(a0Count, tileA0Len);

        OP_LOGI(context, "ARA branch: rMax=%ld tileA0Len=%ld alignedCols=%ld a0Outer=%ld",
                rMax, tileA0Len, alignedCols, a0Outer);

        if (rCount <= rMax) {
            // ARA-全载
            schMode = SCH_ARA_FULLLOAD;
            tmpBufSize = ComputeReduceTmpBufSize(rCount * alignedCols);
        } else {
            // ARA-RowSplit
            schMode = SCH_ARA_ROWSPLIT;
            ComputeAraRowSplitParams(rCount, rMax, rChunkSize, numChunks, lastChunkSize);
            tmpBufSize = ComputeReduceTmpBufSize(rChunkSize * alignedCols);
        }

        // ARA 多核切分：按 A1 × a0Outer 切分
        int64_t totalTiles = a1Count * a0Outer;
        int64_t tilesPerCoreCalc = CeilDiv(totalTiles, coreNum);
        usedCoreNum = CeilDiv(totalTiles, tilesPerCoreCalc);
        tilesPerCore = tilesPerCoreCalc;
        tailCoreTiles = totalTiles - (usedCoreNum - 1) * tilesPerCoreCalc;

        // AtomicAdd 优化：当 totalTiles=1 且 ARA-RowSplit（大R）时，按 R 维度多核切分
        if (totalTiles == 1 && schMode == SCH_ARA_ROWSPLIT && coreNum > 1) {
            // 每核至少处理 rMax 个 R 元素（即1个chunk）
            int64_t maxCores = CeilDiv(rCount, rMax);
            usedCoreNum = coreNum;
            if (usedCoreNum > maxCores) usedCoreNum = maxCores;
            if (usedCoreNum < 1) usedCoreNum = 1;
            int64_t rPerCoreVal = CeilDiv(rCount, usedCoreNum);
            // 对齐到 rMax（确保每核的 R 是 chunk 的整数倍）
            rPerCoreVal = CeilDiv(rPerCoreVal, rMax) * rMax;
            // 重新计算实际使用核数
            usedCoreNum = CeilDiv(rCount, rPerCoreVal);
            if (usedCoreNum < 1) usedCoreNum = 1;
            // 每核处理1个tile（A1=1, a0Outer=1）
            tilesPerCore = 1;
            tailCoreTiles = 1;
            // 记录 AtomicAdd 参数
            tiling->useAtomicAdd = 1;
            tiling->totalRForSplit = rCount;
            tiling->rPerCore = rPerCoreVal;
        }
    }

    OP_LOGI(context, "ReduceNansum schMode=%u usedCoreNum=%ld tilesPerCore=%ld tailCoreTiles=%ld useAtomicAdd=%ld",
            schMode, usedCoreNum, tilesPerCore, tailCoreTiles, tiling->useAtomicAdd);

    // ================================================================
    // Step 4: 填充 TilingData
    // ================================================================
    tiling->a1Count = a1Count;
    tiling->rCount = rCount;
    tiling->a0Count = a0Count;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tilesPerCore = tilesPerCore;
    tiling->tailCoreTiles = tailCoreTiles;
    tiling->tileA0Len = tileA0Len;
    tiling->rChunkSize = rChunkSize;
    tiling->numChunks = numChunks;
    tiling->lastChunkSize = lastChunkSize;
    tiling->rLengthAlign = rLengthAlign;
    tiling->alignedCols = alignedCols;
    tiling->a0Outer = a0Outer;
    tiling->originalA0 = a0Count;
    tiling->tmpBufSize = tmpBufSize;

    // 非连续多轴归约的 strided copy 参数
    if (stridedInfo.isStrided) {
        tiling->copyBlockCount = stridedInfo.blockCount;
        tiling->copyBlockLen = stridedInfo.innerBlockSize * typeSize;
        tiling->copySrcStride = stridedInfo.gapBetweenBlocks * typeSize;
        tiling->outputStride = stridedInfo.outputStride;

        // 非归约维度信息
        tiling->nonReduceDimCount = stridedInfo.nonReduceDimCount;
        for (int64_t i = 0; i < stridedInfo.nonReduceDimCount && i < 8; i++) {
            tiling->nonReduceDimSizes[i] = stridedInfo.nonReduceDimSizes[i];
            tiling->nonReduceGmStrides[i] = stridedInfo.nonReduceGmStrides[i];
        }

        // 归约维度信息
        tiling->reduceDimCount = stridedInfo.reduceDimCount;
        for (int64_t i = 0; i < stridedInfo.reduceDimCount && i < 8; i++) {
            tiling->reduceDimSizes[i] = stridedInfo.reduceDimSizes[i];
            tiling->reduceGmStrides[i] = stridedInfo.reduceGmStrides[i];
        }

        OP_LOGI(context, "Strided copy: blockCount=%ld blockLen=%ld srcStride=%ld outputStride=%ld nrDimCount=%ld rdDimCount=%ld",
                tiling->copyBlockCount, tiling->copyBlockLen, tiling->copySrcStride, tiling->outputStride,
                tiling->nonReduceDimCount, tiling->reduceDimCount);
    }

    context->SetBlockDim(usedCoreNum);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceNansum([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceNansumCompileInfo {};

IMPL_OP_OPTILING(ReduceNansum)
    .Tiling(ReduceNansumTilingFunc)
    .TilingParse<ReduceNansumCompileInfo>(TilingParseForReduceNansum);

} // namespace optiling
