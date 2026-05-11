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
 * \file population_count_tiling.cpp
 * \brief Tiling for PopulationCount (arch35, per DESIGN.md §3.3.2)
 *
 * Strategy:
 *   - Split totalNum across cores; blockFactor aligned to ubBlockSize
 *   - ubFactor = UB / (perElemBytes * bufferMul), aligned to ubBlockSize
 *   - perElemBytes: in (2B) + tmp (2B) + out (1B) = 5 (single buf)
 *                   double buf: in*2 + tmp + out*2 = 8
 *     We use a conservative worst-case total of 8 (double buf) then scale.
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "util/platform_util.h"
#include "../op_kernel/population_count_tiling_data.h"
#include "../op_kernel/population_count_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer threshold: when totalNum > MIN_SPLIT_THRESHOLD switch to BUFFER_MODE=1
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// perElemBytes: x(2B) + tmp(2B) + y(1B) = 5 single; 2*2 + 2 + 1*2 = 8 double (in x2 + tmp not doubled + out x2)
// We divide ubSize by perElemBytes; double buf multiplies in/out queue depth while tmp stays single
constexpr int64_t PER_ELEM_BYTES = 5;        // Single buffer: in(2B) + tmp(2B) + out(1B)
constexpr int64_t PER_ELEM_BYTES_DOUBLE = 8; // Double buffer: in(2B)*2 + tmp(2B) + out(1B)*2

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    totalIdx = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_INT16, ge::DT_UINT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(dataType) == 0,
                OP_LOGE(context, "PopulationCount: invalid input dtype %d (must be INT16/UINT16)",
                        static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PopulationCountTilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Shape / dtype
    int64_t totalIdx = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Empty tensor fast-path (short-circuit BEFORE GetWorkspaceSize and blockFactor division).
    //    Kernel guards with totalNum==0 check; here we only need to:
    //      - zero out TilingData
    //      - set workspace size explicitly to WS_SYS_SIZE (0)
    //      - set block dim to 1
    //      - emit a valid TilingKey via ASCENDC_TPL_SEL_PARAM
    if (totalIdx == 0) {
        PopulationCountTilingData* emptyTiling = context->GetTilingData<PopulationCountTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, emptyTiling);
        OP_CHECK_IF(
            memset_s(emptyTiling, sizeof(PopulationCountTilingData), 0, sizeof(PopulationCountTilingData)) != EOK,
            OP_LOGE(context, "memset tiling data error (empty tensor)"),
            return ge::GRAPH_FAILED);
        emptyTiling->totalNum    = 0;
        emptyTiling->blockFactor = 0;
        emptyTiling->ubFactor    = 0;

        size_t* emptyWorkspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, emptyWorkspace);
        emptyWorkspace[0] = WS_SYS_SIZE;

        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 0U;
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
        return ge::GRAPH_SUCCESS;
    }

    // 4. Workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. TilingData
    PopulationCountTilingData* tiling = context->GetTilingData<PopulationCountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PopulationCountTilingData), 0, sizeof(PopulationCountTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;

    // 6. Multi-core split: align blockFactor to ubBlockSize (DMA min granularity = 32B)
    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, std::max<int64_t>(tiling->blockFactor, 1));

    // 7. UB split
    //    Single buffer: inQ(2B) + tmpBuf(2B) + outQ(1B)    = PER_ELEM_BYTES        (5B/elem)
    //    Double buffer: inQ(2B)*2 + tmpBuf(2B) + outQ(1B)*2 = PER_ELEM_BYTES_DOUBLE (8B/elem)
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t perElemBytes = useDoubleBuffer ? PER_ELEM_BYTES_DOUBLE : PER_ELEM_BYTES;
    int64_t ubFactorRaw = static_cast<int64_t>(ubSize) / perElemBytes;
    tiling->ubFactor = FloorAlign(ubFactorRaw, ubBlockSize);

    context->SetBlockDim(std::max<int64_t>(usedCoreNum, 1));

    // 8. Set TilingKey via template parameters
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPopulationCount([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PopulationCountCompileInfo {}; // required for graph mode

IMPL_OP_OPTILING(PopulationCount)
    .Tiling(PopulationCountTilingFunc)
    .TilingParse<PopulationCountCompileInfo>(TilingParseForPopulationCount);

} // namespace optiling
