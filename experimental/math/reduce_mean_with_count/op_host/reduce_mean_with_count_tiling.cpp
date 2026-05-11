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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file reduce_mean_with_count_tiling.cpp
 * @brief ReduceMeanWithCount tiling implementation (arch35)
 *
 * Iteration 2: TK0 (FP32 AR full-load), TK1 (FP32 AR col-split),
 *              TK2 (FP32 ARA full-load) fully implemented.
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "util/platform_util.h"
#include "reduce_mean_with_count_common.h"
#include "../op_kernel/reduce_mean_with_count_tiling_data.h"
#include "../op_kernel/reduce_mean_with_count_tiling_key.h"
#include <algorithm>
#include <vector>
#include <utility>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using ops_reduce_mean_with_count::IsReduceDim;
using ops_reduce_mean_with_count::NormalizeAxes;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t MIN_OUT_BUF_SIZE = 32;    // Minimum 32 bytes for output buffer
constexpr uint32_t TMP_BUF_DEFAULT = 4096;   // Default tmpBuf size for ReduceSum
constexpr uint32_t ALIGN_32B = 32;
constexpr uint32_t BLOCK_BYTES = 256;        // Repeat block size for ReduceSum tree
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FP16_BYTES = 2;
constexpr int CHUNK_REFINE_ITERS = 4;
constexpr size_t SCENE_SINGLE = 1;
constexpr size_t SCENE_PAIR = 2;
constexpr size_t SCENE_TRIPLE = 3;

struct ShapeLayout {
    uint64_t a1Length = 1;
    uint64_t rLength = 1;
    uint64_t a0Length = 1;
};

struct AlignInfo {
    uint32_t typeSize = FP32_BYTES;
    bool needCast = false;
    uint32_t elemPerBlock = 0;
    uint64_t rLengthAlign = 0;
    uint64_t rLengthAlignFP32 = 0;
    uint64_t tmpBufSize = 0;
};

struct ScheduleInfo {
    uint64_t tilesPerCore = 0;
    uint64_t tailCoreTiles = 0;
    int32_t usedCoreNum = 1;
    uint64_t tileA0Len = 0;
    uint64_t chunkR = 0;
    uint32_t reduceMode = REDUCE_MODE_AR_FULLLOAD;
};

// ============================================================================
// Platform info helpers
// ============================================================================
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Compute tmpBuf size for ReduceSum
// ============================================================================
static uint64_t ComputeReduceBufSize(uint64_t count, uint32_t typeSize)
{
    // ReduceSum Level 2 API temporary buffer size.
    // The internal tree reduction operates on 256-byte blocks and needs space for
    // intermediate results at each level.
    // Empirical formula: max(ceil(count / oneRepeatMaxElem) * oneRepeatMaxElem * typeSize, TMP_BUF_DEFAULT)
    // where oneRepeatMaxElem = 256 / typeSize (e.g., 64 for FP32, 128 for FP16)
    // This gives the tmp buffer enough space for one full repeat of intermediate sums.
    if (typeSize == 0) {
        return TMP_BUF_DEFAULT;
    }
    uint64_t oneRepeatMaxElem = BLOCK_BYTES / typeSize;  // 64 for FP32, 128 for FP16
    if (oneRepeatMaxElem == 0) {
        return TMP_BUF_DEFAULT;
    }
    uint64_t blocks = (count + oneRepeatMaxElem - 1) / oneRepeatMaxElem;
    uint64_t estimated = ((blocks * typeSize + ALIGN_32B - 1) / ALIGN_32B) * ALIGN_32B;
    estimated += BLOCK_BYTES;
    if (estimated < TMP_BUF_DEFAULT) {
        estimated = TMP_BUF_DEFAULT;
    }
    return estimated;
}

// ============================================================================
// Collect shape dims (handle scalar)
// ============================================================================
static void CollectShapeDims(const gert::Shape& inputShape, int64_t rank, std::vector<int64_t>& shapeDims)
{
    shapeDims.assign(rank, 1);
    for (int64_t i = 0; i < rank; i++) {
        shapeDims[i] = (inputShape.GetDimNum() == 0) ? 1 : inputShape.GetDim(i);
    }
}

// ============================================================================
// Compute outputLength
// ============================================================================
static int64_t ComputeOutputLength(
    const std::vector<int64_t>& shapeDims, const std::vector<int64_t>& axes, bool keepdim)
{
    int64_t rank = static_cast<int64_t>(shapeDims.size());
    int64_t outputLength = 1;
    for (int64_t d = 0; d < rank; d++) {
        if (IsReduceDim(axes, d)) {
            if (keepdim) {
                outputLength *= 1;
            }
        } else {
            outputLength *= shapeDims[d];
        }
    }
    if (outputLength == 0) {
        outputLength = 1;
    }
    return outputLength;
}

// ============================================================================
// Axis merging: mark A/R, drop size-1, merge adjacent same-type
// ============================================================================
static std::vector<std::pair<char, int64_t>> MergeAxes(
    const std::vector<int64_t>& shapeDims, const std::vector<int64_t>& axes)
{
    int64_t rank = static_cast<int64_t>(shapeDims.size());
    std::vector<std::pair<char, int64_t>> merged;
    for (int64_t d = 0; d < rank; d++) {
        if (shapeDims[d] == 1 && rank > 1) {
            continue;
        }
        char tag = IsReduceDim(axes, d) ? 'R' : 'A';
        merged.push_back(std::make_pair(tag, shapeDims[d]));
    }
    if (merged.empty()) {
        merged.push_back(std::make_pair('R', static_cast<int64_t>(1)));
    }

    std::vector<std::pair<char, int64_t>> finalMerged;
    finalMerged.push_back(merged[0]);
    for (size_t i = 1; i < merged.size(); i++) {
        if (merged[i].first == finalMerged.back().first) {
            finalMerged.back().second *= merged[i].second;
        } else {
            finalMerged.push_back(merged[i]);
        }
    }
    return finalMerged;
}

// ============================================================================
// Determine scene (A1/R/A0 lengths) from merged groups
// ============================================================================
static void DetermineSceneSingle(
    const std::vector<std::pair<char, int64_t>>& merged, ShapeLayout& layout)
{
    if (merged[0].first == 'R') {
        layout.rLength = merged[0].second;
    } else {
        layout.a1Length = merged[0].second;
    }
}

static void DetermineScenePair(
    const std::vector<std::pair<char, int64_t>>& merged, ShapeLayout& layout)
{
    if (merged[0].first == 'A' && merged[1].first == 'R') {
        layout.a1Length = merged[0].second;
        layout.rLength = merged[1].second;
    } else if (merged[0].first == 'R' && merged[1].first == 'A') {
        layout.rLength = merged[0].second;
        layout.a0Length = merged[1].second;
    } else {
        layout.a1Length = merged[0].second;
        layout.rLength = merged[1].second;
    }
}

static void AbsorbTrailing(
    const std::vector<std::pair<char, int64_t>>& merged, size_t startIdx,
    uint64_t& rLen, uint64_t& a0Len)
{
    for (size_t i = startIdx; i < merged.size(); i++) {
        if (merged[i].first == 'R') {
            rLen *= merged[i].second;
        } else {
            a0Len *= merged[i].second;
        }
    }
}

static void DetermineSceneTripleOrMore(
    const std::vector<std::pair<char, int64_t>>& merged, ShapeLayout& layout)
{
    if (merged[0].first == 'A' && merged[1].first == 'R' && merged[2].first == 'A') {
        layout.a1Length = merged[0].second;
        layout.rLength = merged[1].second;
        layout.a0Length = merged[2].second;
        AbsorbTrailing(merged, SCENE_TRIPLE, layout.rLength, layout.a0Length);
    } else if (merged[0].first == 'R' && merged[1].first == 'A' && merged[2].first == 'R') {
        layout.rLength = merged[0].second * merged[2].second;
        layout.a0Length = merged[1].second;
        AbsorbTrailing(merged, SCENE_TRIPLE, layout.rLength, layout.a0Length);
    } else {
        uint64_t aAcc = 1;
        uint64_t rAcc = 1;
        for (size_t i = 0; i < merged.size(); i++) {
            if (merged[i].first == 'A') {
                aAcc *= merged[i].second;
            } else {
                rAcc *= merged[i].second;
            }
        }
        layout.a1Length = aAcc;
        layout.rLength = rAcc;
    }
}

static ShapeLayout DetermineScene(const std::vector<std::pair<char, int64_t>>& merged)
{
    ShapeLayout layout;
    if (merged.size() == SCENE_SINGLE) {
        DetermineSceneSingle(merged, layout);
    } else if (merged.size() == SCENE_PAIR) {
        DetermineScenePair(merged, layout);
    } else if (merged.size() >= SCENE_TRIPLE) {
        DetermineSceneTripleOrMore(merged, layout);
    }
    return layout;
}

// ============================================================================
// Compute align info (typeSize, alignment, tmpBufSize)
// ============================================================================
static AlignInfo ComputeAlignInfo(ge::DataType dataType, uint64_t rLength)
{
    AlignInfo info;
    info.typeSize = FP32_BYTES;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        info.typeSize = FP16_BYTES;
    }
    info.needCast = (info.typeSize != FP32_BYTES);
    info.elemPerBlock = ALIGN_32B / info.typeSize;
    info.rLengthAlign = CeilAlign(static_cast<int64_t>(rLength),
                                  static_cast<int64_t>(info.elemPerBlock));
    constexpr uint32_t elemPerBlockFP32 = ALIGN_32B / sizeof(float);
    info.rLengthAlignFP32 = CeilAlign(static_cast<int64_t>(rLength),
                                      static_cast<int64_t>(elemPerBlockFP32));
    info.tmpBufSize = ComputeReduceBufSize(
        info.needCast ? info.rLengthAlignFP32 : info.rLengthAlign,
        static_cast<uint32_t>(sizeof(float)));
    return info;
}

// ============================================================================
// AR col-split chunk refinement
// ============================================================================
static void RefineChunkR(
    uint64_t rLength, uint64_t ubSize, const AlignInfo& info,
    uint64_t& chunkR, uint64_t& tmpBufSize)
{
    constexpr uint32_t elemPerBlockFP32 = ALIGN_32B / sizeof(float);
    chunkR = rLength;
    for (int iter = 0; iter < CHUNK_REFINE_ITERS; iter++) {
        uint64_t chunkRAlignFP32 = CeilAlign(static_cast<int64_t>(chunkR),
                                             static_cast<int64_t>(elemPerBlockFP32));
        tmpBufSize = ComputeReduceBufSize(info.needCast ? chunkRAlignFP32 : chunkR,
                                          static_cast<uint32_t>(sizeof(float)));
        uint64_t fixedOverhead = MIN_OUT_BUF_SIZE + tmpBufSize
                               + (info.needCast ? MIN_OUT_BUF_SIZE : 0);
        uint64_t perElemBytes = info.typeSize + (info.needCast ? sizeof(float) : 0);
        uint64_t avail = (ubSize > fixedOverhead) ? (ubSize - fixedOverhead) : 0;
        uint64_t newChunkR = (perElemBytes == 0) ? info.elemPerBlock : (avail / perElemBytes);
        newChunkR = FloorAlign(static_cast<int64_t>(newChunkR),
                               static_cast<int64_t>(info.elemPerBlock));
        if (newChunkR == 0) {
            newChunkR = info.elemPerBlock;
        }
        if (newChunkR > rLength) {
            newChunkR = rLength;
        }
        if (newChunkR == chunkR) {
            chunkR = newChunkR;
            break;
        }
        chunkR = newChunkR;
    }
    uint64_t chunkRAlignFP32Final = CeilAlign(static_cast<int64_t>(chunkR),
                                              static_cast<int64_t>(elemPerBlockFP32));
    tmpBufSize = ComputeReduceBufSize(info.needCast ? chunkRAlignFP32Final : chunkR,
                                      static_cast<uint32_t>(sizeof(float)));
}

// ============================================================================
// AR mode scheduling
// ============================================================================
static void PlanArMode(
    const ShapeLayout& layout, uint64_t ubSize, int64_t coreNum,
    AlignInfo& info, ScheduleInfo& sched)
{
    uint64_t inBufFull = 2 * info.rLengthAlign * info.typeSize;
    uint64_t outBufFull = 2 * MIN_OUT_BUF_SIZE;
    uint64_t castBufFull = info.needCast ? (info.rLengthAlignFP32 * sizeof(float)) : 0;
    uint64_t fp32ResBufFull = info.needCast ? MIN_OUT_BUF_SIZE : 0;
    uint64_t ubNeeded = inBufFull + outBufFull + info.tmpBufSize + castBufFull + fp32ResBufFull;

    if (ubNeeded <= ubSize) {
        sched.reduceMode = REDUCE_MODE_AR_FULLLOAD;
    } else {
        sched.reduceMode = REDUCE_MODE_AR_COLSPLIT;
        RefineChunkR(layout.rLength, ubSize, info, sched.chunkR, info.tmpBufSize);
    }

    uint64_t rowsPerCore = CeilDiv(static_cast<int64_t>(layout.a1Length), coreNum);
    if (rowsPerCore == 0) {
        rowsPerCore = 1;
    }
    sched.usedCoreNum = static_cast<int32_t>(CeilDiv(
        static_cast<int64_t>(layout.a1Length), static_cast<int64_t>(rowsPerCore)));
    sched.tilesPerCore = rowsPerCore;
    sched.tailCoreTiles = layout.a1Length - sched.tilesPerCore * (sched.usedCoreNum - 1);
}

// ============================================================================
// ARA mode: find best tileA0
// ============================================================================
static uint64_t FindBestTileA0(
    const ShapeLayout& layout, uint64_t ubSize, const AlignInfo& info, uint32_t a0TileBase)
{
    uint64_t bestTileA0 = a0TileBase;
    for (uint64_t candidate = a0TileBase; candidate <= layout.a0Length; candidate += a0TileBase) {
        uint64_t candidateBytes = candidate * info.typeSize;
        uint64_t alignedBytes = ((candidateBytes + ALIGN_32B - 1) / ALIGN_32B) * ALIGN_32B;
        uint64_t alignedCols = alignedBytes / info.typeSize;

        uint64_t inBufSize = 2ULL * layout.rLength * alignedCols * info.typeSize;
        uint64_t outBufElemBytes = ((alignedCols * info.typeSize + ALIGN_32B - 1) / ALIGN_32B) * ALIGN_32B;
        uint64_t outBufSize = 2ULL * outBufElemBytes;
        uint64_t castBufSize = info.needCast ? (layout.rLength * alignedCols * sizeof(float)) : 0;
        uint64_t fp32ResBufSize = info.needCast
            ? (((alignedCols * sizeof(float) + ALIGN_32B - 1) / ALIGN_32B) * ALIGN_32B)
            : 0;

        uint64_t minCountElems = (candidate < 4) ? 4 : candidate;
        uint64_t countBufSize = ((minCountElems * sizeof(int64_t) + ALIGN_32B - 1) / ALIGN_32B) * ALIGN_32B;

        uint64_t totalUB = inBufSize + outBufSize + info.tmpBufSize
                         + castBufSize + fp32ResBufSize + countBufSize;
        if (totalUB <= ubSize) {
            bestTileA0 = candidate;
        } else {
            break;
        }
    }
    return bestTileA0;
}

static void PlanAraMode(
    const ShapeLayout& layout, uint64_t ubSize, int64_t coreNum,
    const AlignInfo& info, ScheduleInfo& sched)
{
    sched.reduceMode = REDUCE_MODE_ARA_FULLLOAD;
    uint32_t a0TileBase = ALIGN_32B / info.typeSize;
    uint64_t bestTileA0 = FindBestTileA0(layout, ubSize, info, a0TileBase);
    sched.tileA0Len = bestTileA0;

    if (sched.tileA0Len > layout.a0Length) {
        sched.tileA0Len = ((layout.a0Length + a0TileBase - 1) / a0TileBase) * a0TileBase;
        if (sched.tileA0Len > layout.a0Length) {
            sched.tileA0Len = layout.a0Length;
        }
    }
    if (sched.tileA0Len == 0) {
        sched.tileA0Len = a0TileBase;
    }

    uint64_t a0Outer = CeilDiv(static_cast<int64_t>(layout.a0Length),
                               static_cast<int64_t>(sched.tileA0Len));
    uint64_t totalTiles = layout.a1Length * a0Outer;
    sched.tilesPerCore = CeilDiv(static_cast<int64_t>(totalTiles), coreNum);
    if (sched.tilesPerCore == 0) {
        sched.tilesPerCore = 1;
    }
    sched.usedCoreNum = static_cast<int32_t>(CeilDiv(
        static_cast<int64_t>(totalTiles), static_cast<int64_t>(sched.tilesPerCore)));
    sched.tailCoreTiles = totalTiles - sched.tilesPerCore * (sched.usedCoreNum - 1);
}

// ============================================================================
// Write out TilingData
// ============================================================================
static ge::graphStatus FillTilingData(
    gert::TilingContext* context, const ShapeLayout& layout, const AlignInfo& info,
    const ScheduleInfo& sched, float invCount, int64_t countResult, int64_t outputLength)
{
    ReduceMeanWithCountTilingData* tiling = context->GetTilingData<ReduceMeanWithCountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReduceMeanWithCountTilingData), 0, sizeof(ReduceMeanWithCountTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->a1Length = layout.a1Length;
    tiling->rLength = layout.rLength;
    tiling->a0Length = layout.a0Length;
    tiling->usedCoreNum = sched.usedCoreNum;
    tiling->tilesPerCore = sched.tilesPerCore;
    tiling->tailCoreTiles = sched.tailCoreTiles;
    tiling->tileA0Len = sched.tileA0Len;
    tiling->chunkR = sched.chunkR;
    tiling->invCount = invCount;
    tiling->countResult = countResult;
    tiling->tmpBufSize = info.tmpBufSize;
    tiling->outputLength = static_cast<uint64_t>(outputLength);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Parsed inputs package
// ============================================================================
struct ParsedInputs {
    gert::Shape inputShape;
    ge::DataType dataType = ge::DT_FLOAT;
    int64_t rank = 1;
    std::vector<int64_t> shapeDims;
    std::vector<int64_t> axes;
    bool keepdim = false;
    int64_t countResult = 1;
    float invCount = 1.0f;
    int64_t outputLength = 1;
};

static ge::graphStatus ParseInputs(gert::TilingContext* context, ParsedInputs& parsed)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    parsed.inputShape = inputShapePtr->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    parsed.dataType = inputDesc->GetDataType();

    parsed.rank = static_cast<int64_t>(parsed.inputShape.GetDimNum());
    if (parsed.rank == 0) {
        parsed.rank = 1;
    }
    CollectShapeDims(parsed.inputShape, parsed.rank, parsed.shapeDims);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    parsed.axes = NormalizeAxes(attrs->GetListInt(0), parsed.rank);

    parsed.countResult = 1;
    for (size_t a = 0; a < parsed.axes.size(); a++) {
        parsed.countResult *= parsed.shapeDims[parsed.axes[a]];
    }
    OP_CHECK_IF(parsed.countResult == 0,
        OP_LOGE(context, "reduce count is 0 (empty tensor on reduce axis is not supported)"),
        return ge::GRAPH_FAILED);
    parsed.invCount = 1.0f / static_cast<float>(parsed.countResult);

    const bool* keepdimPtr = attrs->GetBool(1);
    parsed.keepdim = (keepdimPtr != nullptr) ? *keepdimPtr : false;
    parsed.outputLength = ComputeOutputLength(parsed.shapeDims, parsed.axes, parsed.keepdim);
    return ge::GRAPH_SUCCESS;
}

static void PlanSchedule(
    const ParsedInputs& parsed, uint64_t ubSize, int64_t coreNum,
    ShapeLayout& layout, AlignInfo& info, ScheduleInfo& sched)
{
    auto finalMerged = MergeAxes(parsed.shapeDims, parsed.axes);
    layout = DetermineScene(finalMerged);
    info = ComputeAlignInfo(parsed.dataType, layout.rLength);
    if (layout.a0Length == 1) {
        PlanArMode(layout, ubSize, coreNum, info, sched);
    } else {
        PlanAraMode(layout, ubSize, coreNum, info, sched);
    }
}

// ============================================================================
// Main tiling function
// ============================================================================
static ge::graphStatus ReduceMeanWithCountTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    ParsedInputs parsed;
    auto parseSt = ParseInputs(context, parsed);
    if (parseSt != ge::GRAPH_SUCCESS) {
        return parseSt;
    }

    ShapeLayout layout;
    AlignInfo info;
    ScheduleInfo sched;
    PlanSchedule(parsed, ubSize, coreNum, layout, info, sched);

    auto st = FillTilingData(context, layout, info, sched, parsed.invCount,
                             parsed.countResult, parsed.outputLength);
    if (st != ge::GRAPH_SUCCESS) {
        return st;
    }

    context->SetBlockDim(sched.usedCoreNum);
    uint32_t dTypeX = static_cast<uint32_t>(parsed.dataType);
    uint32_t reduceMode = sched.reduceMode;
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, reduceMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceMeanWithCount([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceMeanWithCountCompileInfo {};

IMPL_OP_OPTILING(ReduceMeanWithCount)
    .Tiling(ReduceMeanWithCountTilingFunc)
    .TilingParse<ReduceMeanWithCountCompileInfo>(TilingParseForReduceMeanWithCount);

} // namespace optiling
