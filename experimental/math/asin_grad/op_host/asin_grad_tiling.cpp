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

/*!
 * \file asin_grad_tiling.cpp
 * \brief AsinGrad Tiling implementation (arch35 / Ascend950)
 *
 * Tiling strategy:
 *   1. Multi-core: evenly distribute elements across cores, aligned to UB block size
 *   2. UB split: divide per-core elements into UB-sized chunks
 *   3. Buffer mode: double buffer for large data, single buffer for small data
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_host/util/math_util.h"
#include "util/platform_util.h"
#include "../op_kernel/asin_grad_tiling_data.h"
#include "../op_kernel/asin_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer threshold: enable double buffer when data > this value
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    // Get input dy shape
    auto inputDy = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDy);
    auto dyShape = EnsureNotScalar(inputDy->GetStorageShape());

    // Get input x shape
    auto inputX = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xShape = EnsureNotScalar(inputX->GetStorageShape());

    // Get output dx shape
    auto outDx = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDx);
    auto dxShape = EnsureNotScalar(outDx->GetStorageShape());

    // Shape validation: dy, x, dx must have same shape
    OP_CHECK_IF(
        dyShape.GetShapeSize() != xShape.GetShapeSize() ||
            dyShape.GetShapeSize() != dxShape.GetShapeSize(),
        OP_LOGE(context, "AsinGrad: shape size mismatch: dy=%ld, x=%ld, dx=%ld",
                dyShape.GetShapeSize(), xShape.GetShapeSize(), dxShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = dyShape.GetShapeSize();

    // Dtype validation
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AsinGrad: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static int64_t CalcUbFactor(ge::DataType dataType, uint64_t ubSize, int64_t ubBlockSize, uint64_t useDoubleBuffer)
{
    if (dataType == ge::DT_BF16) {
        // bf16 path: TQue for bf16 I/O + TBuf<VECCALC> for float intermediate compute
        // TQue: (2 bf16 input + 1 bf16 output) queues, each with BUFFER_NUM slots
        //   = 3 * sizeof(bf16) * BUFFER_NUM = 3 * 2 * BUFFER_NUM
        // TBuf: 4 float buffers (dyF32, xF32, tmpF32, dxF32), no double-buffer
        //   = 4 * sizeof(float) = 4 * 4 = 16
        // Total per element:
        //   single buffer: 3*2*1 + 16 = 22 bytes
        //   double buffer: 3*2*2 + 16 = 28 bytes
        int64_t bytesPerElement = useDoubleBuffer ? 28 : 22;
        return FloorAlign(FloorDiv(static_cast<int64_t>(ubSize), bytesPerElement), ubBlockSize);
    } else {
        // fp16/fp32 path: 4 buffers (dy, x, tmp, dx)
        int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
        int64_t bufferNum = useDoubleBuffer ? 8 : 4;
        return FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferNum), ubBlockSize);
    }
}

static ge::graphStatus AsinGradTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2. Get shape and attr info
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);
    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    // Handle empty tensor
    if (totalNum == 0) {
        context->SetBlockDim(0);
        AsinGradTilingData* tiling = context->GetTilingData<AsinGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(AsinGradTilingData), 0, sizeof(AsinGradTilingData));
        // Still need to set TilingKey for empty case
        uint32_t dType = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 0;
        ASCENDC_TPL_SEL_PARAM(context, dType, useDoubleBuffer);
        return ge::GRAPH_SUCCESS;
    }
    // 4. Set tiling data
    AsinGradTilingData* tiling = context->GetTilingData<AsinGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AsinGradTilingData), 0, sizeof(AsinGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);
    // Determine buffer mode
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    tiling->ubFactor = CalcUbFactor(dataType, ubSize, ubBlockSize, useDoubleBuffer);

    context->SetBlockDim(usedCoreNum);
    // 5. Set TilingKey
    uint32_t dType = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dType, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAsinGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AsinGradCompileInfo {};

IMPL_OP_OPTILING(AsinGrad).Tiling(AsinGradTilingFunc).TilingParse<AsinGradCompileInfo>(TilingParseForAsinGrad);

} // namespace optiling
