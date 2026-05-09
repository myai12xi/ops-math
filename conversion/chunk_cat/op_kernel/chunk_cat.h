/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file chunk_cat.h
 * \brief
 */

#ifndef _CHUNK_CAT_DATA_H_
#define _CHUNK_CAT_DATA_H_

#include "chunk_cat_common.h"

using namespace AscendC;
template <typename T1, typename T2, bool NEAD_CAST=false>
class ChunkCat
{
public:
    __aicore__ inline ChunkCat(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ChunkCatTilingData& tilingData)
    {
        blockIdx_ = GetBlockIdx();
        int32_t usedCoreNum = GetBlockNum();
        // 获取tiling信息
        isAllAlign_ = tilingData.isAllAlign;
        isHalfAlign_ = tilingData.isHalfAlign;
        isOneConcat_ = tilingData.isOneConcat;
        dim_ = tilingData.dim;
        numChunk_ = tilingData.numChunk;
        outputRow_ = tilingData.outputRow;
        outputCol_ = tilingData.outputCol;
        blockRowFactor_ = tilingData.blockRowFactor;
        blockColFactor_ = tilingData.blockColFactor;
        tailBlockRowFactor_ = tilingData.tailBlockRowFactor;
        tailBlockColFactor_ = tilingData.tailBlockColFactor;
        ubRowFactor_ = tilingData.ubRowFactor;
        ubColFactor_ = tilingData.ubColFactor;
        inputNum_ = tilingData.inputNum;
        srcEleUbBlock_ = UB_BLOCK_SIZE / sizeof(T1);
        dstEleUbBlock_ = UB_BLOCK_SIZE / sizeof(T2);
        colRepeatNum_ = isHalfAlign_ ? HALF : srcEleUbBlock_;

        blockRowGroup_ = blockIdx_ / tilingData.blockColNum;
        blockColGroup_ = blockIdx_ % tilingData.blockColNum;
        currentBlockRowFactor_ = blockRowGroup_ == tilingData.blockRowNum - 1 ? tailBlockRowFactor_ : blockRowFactor_;
        currentBlockColFactor_ = blockColGroup_ == tilingData.blockColNum - 1 ? tailBlockColFactor_ : blockColFactor_;
        int64_t dstGmOffset = blockRowGroup_ * blockRowFactor_ * outputCol_ + blockColGroup_ * blockColFactor_;
        dstGlobal_.SetGlobalBuffer((__gm__ T2*)y + dstGmOffset);
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));

        pipe_->InitBuffer(srcBuf_, tilingData.inUbSize);
        pipe_->InitBuffer(dstBuf_, tilingData.outUbSize);
        srcLocal_ = srcBuf_.Get<T1>();
        srcLocalT2_ = srcLocal_.template ReinterpretCast<T2>();
        dstLocal_ = dstBuf_.Get<T2>();
        dstLocalT1_ = dstLocal_.template ReinterpretCast<T1>();
        dstLocalFP32_ = dstLocal_.template ReinterpretCast<float>();
    }

    __aicore__ inline void Process()
    {
        int64_t rowLoop = GetAlign(currentBlockRowFactor_, ubRowFactor_) / ubRowFactor_;
        int64_t colLoop = GetAlign(currentBlockColFactor_, ubColFactor_) / ubColFactor_;
        int64_t rowTail = currentBlockRowFactor_ % ubRowFactor_;
        int64_t colTail = currentBlockColFactor_ % ubColFactor_;

        uint64_t buf[10];
        desc_.SetShapeAddr(buf); // 用于获取shape信息
        int64_t inputCol[32];
        
        for (int64_t i = 0; i < rowLoop * colLoop; i++) {
            UbLoopInfo ubLoopInfo{};
            ubLoopInfo.inputCol = inputCol;
            ubLoopInfo.ubRowGroup = i / colLoop;
            ubLoopInfo.ubColGroup = i % colLoop;
            ubLoopInfo.currentUbRowFactor = (rowTail != 0 && ubLoopInfo.ubRowGroup == rowLoop - 1) ?
                                            rowTail : ubRowFactor_;
            ubLoopInfo.currentUbColFactor = (colTail != 0 && ubLoopInfo.ubColGroup == colLoop - 1) ?
                                            colTail : ubColFactor_;
            // 搬入
            CopyIn(ubLoopInfo);
            // 计算
            Compute(ubLoopInfo);
            // 搬出
            CopyCout(ubLoopInfo);
        }
    }

private:
    __aicore__ inline void CopyIn(UbLoopInfo& ubLoopInfo)
    {
        // 1、清零ub
        dupToZero();
        // 2、遍历tensor搬运
        int64_t totalCol = 0;
        int64_t localOffset = 0;
        ubLoopInfo.colStart = blockColGroup_ * blockColFactor_ + ubLoopInfo.ubColGroup * ubColFactor_;
        ubLoopInfo.rowStart = blockRowGroup_ * blockRowFactor_ + ubLoopInfo.ubRowGroup * ubRowFactor_;

        for (uint32_t i = 0; i < inputNum_; i++) {
            if (ubLoopInfo.totalUbCol >= ubLoopInfo.currentUbColFactor) {
                break;
            }
            srcGlobal_.SetGlobalBuffer(inputList_.GetDataPtr<T1>(i));
            TensorInfo tensorInfo{};
            inputList_.GetDesc(desc_, i); // scalar很大(将buf改为局部变量有改善)
            // 获取chunk相关信息
            tensorInfo.chunkDimSize = desc_.GetShape(dim_);
            tensorInfo.chunkCol = (tensorInfo.chunkDimSize + numChunk_ - 1) / numChunk_;
            CopyInChunk(totalCol, localOffset, ubLoopInfo, tensorInfo);

            if (isOneConcat_ && ubLoopInfo.count > 31) {
                // 计算
                Compute(ubLoopInfo);
                // 搬出
                CopyCout(ubLoopInfo);
                localOffset = 0;
                ubLoopInfo.preCatCol += ubLoopInfo.totalUbCol;
                ubLoopInfo.count = 0;
                ubLoopInfo.totalUbCol = 0;
                ubLoopInfo.totalUbColAlign = 0;
            }
            else if (ubLoopInfo.count > 31) {
                // 提前做部分concat
                if (!ubLoopInfo.isAllZero) {
                    SetFlag<HardEvent::MTE2_V>(event_);
                    WaitFlag<HardEvent::MTE2_V>(event_);
                    if (isAllAlign_) {
                        UBRearrange4Concat(ubLoopInfo, srcLocal_, dstLocalT1_);
                        DataCopy(srcLocal_, dstLocalT1_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbCol);
                    } else {
                        // 3、ub重排
                        UBRearrange4Trans(ubLoopInfo, srcLocal_, dstLocalT1_);
                        // 4、跨block对齐转置
                        Trans1(ubLoopInfo, dstLocalT1_, srcLocal_);
                        // 5、ub重排
                        UBRearrange4TransConcat<true>(ubLoopInfo, srcLocal_, dstLocalT1_);
                        // 6、跨block对齐转置
                        Trans2<true>(ubLoopInfo, dstLocalT1_, srcLocal_);
                    }
                    SetFlag<HardEvent::V_MTE2>(event_);
                    WaitFlag<HardEvent::V_MTE2>(event_);
                }
                ubLoopInfo.inputCol[0] = ubLoopInfo.totalUbCol;
                ubLoopInfo.count = 1;
                ubLoopInfo.totalUbColAlign = ubLoopInfo.totalUbCol;
                localOffset = isAllAlign_ ? ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbCol :
                                         (isHalfAlign_ ? TRANS_BLOCK * HALF * ubLoopInfo.totalUbCol :
                                                         TRANS_BLOCK * srcEleUbBlock_ * ubLoopInfo.totalUbCol);
            }
        }
    }

    __aicore__ inline void Compute(const UbLoopInfo& ubLoopInfo)
    {
        if (isOneConcat_) {
            ComputeOneConcat(ubLoopInfo);
        }
        else if (ubLoopInfo.isAllZero) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE3_V>(event_);
            WaitFlag<HardEvent::MTE3_V>(event_);
            if constexpr (NEAD_CAST) {
                uint32_t castCount = ubLoopInfo.currentUbRowFactor * GetAlign(ubLoopInfo.currentUbColFactor, dstEleUbBlock_);
                DoCast(ubLoopInfo, castCount);
            } else {
                DataCopy(dstLocalT1_, srcLocal_, ubLoopInfo.currentUbRowFactor * GetAlign(ubLoopInfo.currentUbColFactor, dstEleUbBlock_));
            }
        }
        else if (ubLoopInfo.count == 1 && ubLoopInfo.currentUbColFactor % srcEleUbBlock_ == 0) {
            SetFlag<HardEvent::MTE2_V>(event_);
            WaitFlag<HardEvent::MTE2_V>(event_);
            if constexpr (NEAD_CAST) {
                uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor;
                DoCast(ubLoopInfo, castCount);
            } else {
                DataCopy(dstLocalT1_, srcLocal_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor);
            }
        }
        else if (isAllAlign_) {
            ComputeAllAlign(ubLoopInfo);
        } else {
            ComputeNotAlign(ubLoopInfo);
        }
        SetFlag<HardEvent::V_MTE3>(event_);
        WaitFlag<HardEvent::V_MTE3>(event_);
    }

    __aicore__ inline void CopyCout(const UbLoopInfo& ubLoopInfo)
    {
        if (isOneConcat_) {
            int64_t localOffset = 0;
            int64_t globalOffset = ubLoopInfo.ubRowGroup * ubRowFactor_ * outputCol_ + ubLoopInfo.ubColGroup * ubColFactor_ + ubLoopInfo.preCatCol;
            for (int i = 0; i < ubLoopInfo.count; i++) {
                uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
                uint32_t blockLen = ubLoopInfo.inputCol[i] * sizeof(T2);
                uint32_t dstStride = (outputCol_ - ubLoopInfo.inputCol[i]) * sizeof(T2);
                DataCopyExtParams copyParamsOut{blockCount, blockLen, 0, dstStride, 0};
                DataCopyPad(dstGlobal_[globalOffset], dstLocal_[localOffset], copyParamsOut);
                localOffset += GetAlign(ubLoopInfo.inputCol[i], srcEleUbBlock_);
                globalOffset += ubLoopInfo.inputCol[i];
            }
            SetFlag<HardEvent::MTE3_MTE2>(event_);
            WaitFlag<HardEvent::MTE3_MTE2>(event_);
            return;
        }
        uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
        uint32_t blockLen = ubLoopInfo.currentUbColFactor * sizeof(T2);
        uint32_t dstStride = (outputCol_ - ubLoopInfo.currentUbColFactor)* sizeof(T2);
        DataCopyExtParams copyParamsOut{blockCount, blockLen, 0, dstStride, 0};
        int64_t dstOffset = ubLoopInfo.ubRowGroup * ubRowFactor_ * outputCol_ + ubLoopInfo.ubColGroup * ubColFactor_;
        DataCopyPad(dstGlobal_[dstOffset], dstLocal_, copyParamsOut);
        SetFlag<HardEvent::MTE3_MTE2>(event_);
        WaitFlag<HardEvent::MTE3_MTE2>(event_);
    }

    __aicore__ inline int64_t GetAlign(int64_t value, int64_t align)
    {
        return align == 0 ? value : (value + align - 1) / align * align;
    }

    __aicore__ inline void dupToZero()
    {
        T1 inputVal(0.0);
        Duplicate<T1>(srcLocal_, inputVal, srcLocal_.GetSize());
        SetFlag<HardEvent::V_MTE2>(event_);
        WaitFlag<HardEvent::V_MTE2>(event_);
    }

    __aicore__ inline bool IsTensorInRange(int64_t totalCol, const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo)
    {
        return (totalCol < ubLoopInfo.colStart + ubLoopInfo.currentUbColFactor) &&
               (totalCol + tensorInfo.tensorCol > ubLoopInfo.colStart);
    }

    __aicore__ inline void SplitTensorDim0(int64_t& totalCol, const UbLoopInfo& ubLoopInfo, TensorInfo& tensorInfo)
    {
        // tensor是否被切分
        tensorInfo.splitCol = tensorInfo.tensorCol;
        int64_t colEnd = ubLoopInfo.colStart + ubLoopInfo.currentUbColFactor;
        if (totalCol < ubLoopInfo.colStart && (totalCol + tensorInfo.tensorCol) > colEnd) {
            // 中间部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = ubLoopInfo.currentUbColFactor;
            tensorInfo.startOffset = ubLoopInfo.colStart - totalCol;
        } else if (totalCol < ubLoopInfo.colStart) {
            // 被切分的后半部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = totalCol + tensorInfo.tensorCol - ubLoopInfo.colStart;
            tensorInfo.startOffset = ubLoopInfo.colStart - totalCol;
        } else if ((totalCol + tensorInfo.tensorCol) > colEnd) {
            // 被切分的前半部分
            tensorInfo.isSplit = true;
            tensorInfo.splitCol = colEnd - totalCol;
        }
        tensorInfo.splitColAlign = GetAlign(tensorInfo.splitCol, srcEleUbBlock_);
    }

    __aicore__ inline void ExecuteDataCopy(int64_t localOffset, int64_t gmOffset, uint16_t blockCount,
                                           uint32_t blockLen, uint32_t srcStride)
    {
        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, 0, 0};
        uint8_t rightPadValue = (GetAlign(blockLen, UB_BLOCK_SIZE) - blockLen) / sizeof(T1);
        AscendC::DataCopyPadExtParams<T1> padParams{true, 0, rightPadValue, 0};
        AscendC::DataCopyPad(srcLocal_[localOffset], srcGlobal_[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void DoRowsCopy(int64_t localOffset, const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo)
    {
        uint16_t blockCount = tensorInfo.isSplit ? static_cast<uint16_t>(ubLoopInfo.currentUbRowFactor) : 1;
        uint32_t blockLen = tensorInfo.isSplit ?
            static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1)) :
            static_cast<uint32_t>(ubLoopInfo.currentUbRowFactor * tensorInfo.splitCol * sizeof(T1));
        uint32_t srcStride = (tensorInfo.tensorCol - tensorInfo.splitCol) * sizeof(T1);
        int64_t gmOffset = ubLoopInfo.rowStart * tensorInfo.tensorCol + tensorInfo.startOffset;
        ExecuteDataCopy(localOffset, gmOffset, blockCount, blockLen, srcStride);
    }

    __aicore__ inline void DoLastRowsCopy(int64_t localOffset, const UbLoopInfo& ubLoopInfo, const TensorInfo& tensorInfo)
    {
        // 0 无切分
        int64_t srcGmOffset = ubLoopInfo.rowStart * tensorInfo.tensorCol;
        uint32_t srcStride = (tensorInfo.tensorCol - tensorInfo.splitCol) * sizeof(T1);
        if (!tensorInfo.isSplit) {
            uint32_t blockLen = static_cast<uint32_t>(
                (tensorInfo.chunkDimSize * tensorInfo.originCol - ubLoopInfo.rowStart * tensorInfo.tensorCol) * sizeof(T1));
            ExecuteDataCopy(localOffset, srcGmOffset + tensorInfo.startOffset, 1, blockLen, srcStride);
            return;
        }

        uint16_t blockCount = 0;
        uint32_t blockLen = 0;
        int64_t remainderCol = (tensorInfo.chunkDimSize % tensorInfo.chunkCol) * tensorInfo.originCol;
        // 1 有切分
        // 1.0 remainder等于0
        if (remainderCol == 0) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.1 切分+偏移值小于等于remainder
        else if (tensorInfo.startOffset + tensorInfo.splitCol <= remainderCol) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRowAlign - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.2 偏移值大于等于remainder
        else if (tensorInfo.startOffset >= remainderCol) {
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        // 1.3  偏移值小于remainder，且切分+偏移值大于remainder
        else {
            // 1.3.1
            blockLen = (remainderCol - tensorInfo.startOffset) * sizeof(T1);
            int64_t localOffsetPart = localOffset + (tensorInfo.chunkRow - ubLoopInfo.rowStart) * tensorInfo.splitColAlign;
            int64_t gmOffsetPart = srcGmOffset + tensorInfo.startOffset +
                                   (tensorInfo.chunkRow - ubLoopInfo.rowStart) * tensorInfo.tensorCol;
            ExecuteDataCopy(localOffsetPart, gmOffsetPart, 1, blockLen, srcStride);
            // 1.3.2
            blockCount = static_cast<uint16_t>(tensorInfo.chunkRow - ubLoopInfo.rowStart);
            blockLen = static_cast<uint32_t>(tensorInfo.splitCol * sizeof(T1));
        }
        ExecuteDataCopy(localOffset, srcGmOffset + tensorInfo.startOffset, blockCount, blockLen, srcStride);
    }

    __aicore__ inline void CopyInChunk(int64_t& totalCol, int64_t& localOffset, UbLoopInfo& ubLoopInfo, TensorInfo& tensorInfo)
    {
        // 获取concat阶段输入的col
        for (uint32_t j = 1; j < desc_.GetDim(); j++) {
            tensorInfo.originCol *= desc_.GetShape(j);
        }
        tensorInfo.tensorCol = tensorInfo.chunkCol * tensorInfo.originCol;
        // 判断当前核是否处理当前tensor
        if (!IsTensorInRange(totalCol, ubLoopInfo, tensorInfo)) {
            totalCol += tensorInfo.tensorCol;
            return;
        }
        SplitTensorDim0(totalCol, ubLoopInfo, tensorInfo);
        tensorInfo.chunkRow = tensorInfo.chunkDimSize / tensorInfo.chunkCol;
        tensorInfo.chunkRowAlign = GetAlign(tensorInfo.chunkDimSize, tensorInfo.chunkCol) / tensorInfo.chunkCol;
        int64_t localOffsetIncrement = (isOneConcat_ || isAllAlign_) ? ubLoopInfo.currentUbRowFactor :
            (isHalfAlign_ ? TRANS_BLOCK * HALF : TRANS_BLOCK * srcEleUbBlock_);
        if (ubLoopInfo.rowStart >= tensorInfo.chunkRowAlign) {
            ubLoopInfo.inputCol[ubLoopInfo.count] = tensorInfo.splitCol;
            ubLoopInfo.totalUbColAlign += tensorInfo.splitCol;
            ubLoopInfo.totalUbCol += tensorInfo.splitCol;
            ubLoopInfo.count++;
            totalCol += tensorInfo.tensorCol;
            localOffset += localOffsetIncrement * tensorInfo.splitCol;
            return;
        }
        ubLoopInfo.isAllZero = false;
        
        int64_t rowEnd = ubLoopInfo.rowStart + ubLoopInfo.currentUbRowFactor;
        if (rowEnd < tensorInfo.chunkRowAlign) {
            DoRowsCopy(localOffset, ubLoopInfo, tensorInfo);
        } else {
            DoLastRowsCopy(localOffset, ubLoopInfo, tensorInfo);
        }
        
        localOffsetIncrement *= (isOneConcat_ || tensorInfo.isSplit) ? tensorInfo.splitColAlign : tensorInfo.splitCol;
        if (isOneConcat_) {
            ubLoopInfo.inputCol[ubLoopInfo.count] = tensorInfo.splitCol;
            ubLoopInfo.totalUbColAlign += tensorInfo.splitColAlign;
        } else if (isAllAlign_) {
            ubLoopInfo.inputCol[ubLoopInfo.count] = tensorInfo.splitCol;
            ubLoopInfo.totalUbColAlign += tensorInfo.splitCol;
        } else if (tensorInfo.isSplit) {
            ubLoopInfo.inputCol[ubLoopInfo.count] = -tensorInfo.splitCol;
            ubLoopInfo.totalUbColAlign += tensorInfo.splitColAlign;
        } else {
            ubLoopInfo.inputCol[ubLoopInfo.count] = tensorInfo.splitCol;
            ubLoopInfo.totalUbColAlign += tensorInfo.splitCol;
        }
        ubLoopInfo.totalUbCol += tensorInfo.splitCol;
        ubLoopInfo.count++;
        totalCol += tensorInfo.tensorCol;
        localOffset += localOffsetIncrement;
    }

    __aicore__ inline void UBRearrange4Trans(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = TRANS_BLOCK;
            uint16_t actualCol = ubLoopInfo.inputCol[i] > 0 ? ubLoopInfo.inputCol[i] :
                                 GetAlign(-ubLoopInfo.inputCol[i], srcEleUbBlock_);
            uint16_t blockLen = actualCol * colRepeatNum_ / srcEleUbBlock_;
            uint16_t dstGap = ubLoopInfo.totalUbColAlign * colRepeatNum_ / srcEleUbBlock_ - blockLen;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += blockCount * blockLen * srcEleUbBlock_;
            dstOffset += blockLen * srcEleUbBlock_;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Trans1(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        uint8_t repeatTimes = ubLoopInfo.totalUbColAlign * colRepeatNum_ / srcEleUbBlock_;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        
        uint64_t srcLocalList[TRANS_BLOCK];
        uint64_t dstLocalList[TRANS_BLOCK];
        if constexpr (sizeof(T1) == 2) {
            LocalTensor<half> srcLocalFP16 = srcLocal.template ReinterpretCast<half>();
            LocalTensor<half> dstLocalFP16 = dstLocal.template ReinterpretCast<half>();
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * ubLoopInfo.totalUbColAlign * colRepeatNum_;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocalFP16[offset].GetPhyAddr());
            }
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * TRANS_BLOCK;
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocalFP16[offset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        } else {
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * ubLoopInfo.totalUbColAlign * colRepeatNum_;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            for (uint64_t i = 0; i < srcEleUbBlock_; i++) {
                for (uint64_t j = 0; j < TRANS_BLOCK / srcEleUbBlock_; j++) {
                    uint64_t offset = i * TRANS_BLOCK + j * srcEleUbBlock_;
                    dstLocalList[i * TRANS_BLOCK / srcEleUbBlock_ + j] =
                        reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool NO_NEED_ALIGN=false>
    __aicore__ inline void UBRearrange4TransConcat(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = colRepeatNum_;
            uint16_t actualCol = ubLoopInfo.inputCol[i] > 0 ? ubLoopInfo.inputCol[i] : -ubLoopInfo.inputCol[i];
            uint16_t blockLen = actualCol * TRANS_BLOCK / srcEleUbBlock_;
            uint16_t srcGap = ubLoopInfo.inputCol[i] > 0 ? 0 :
                (GetAlign(-ubLoopInfo.inputCol[i], srcEleUbBlock_) + ubLoopInfo.inputCol[i]) * TRANS_BLOCK / srcEleUbBlock_;
            uint16_t dstGap = GetAlign(ubLoopInfo.totalUbCol, dstEleUbBlock_) * TRANS_BLOCK / srcEleUbBlock_ - blockLen;
            if constexpr (NO_NEED_ALIGN) {
                dstGap = ubLoopInfo.totalUbCol * TRANS_BLOCK / srcEleUbBlock_ - blockLen;
            }
            DataCopyParams copyParams{blockCount, blockLen, srcGap, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += ubLoopInfo.inputCol[i] > 0 ? (colRepeatNum_ * actualCol * TRANS_BLOCK) :
                         (colRepeatNum_ * GetAlign(actualCol, srcEleUbBlock_) * TRANS_BLOCK);
            dstOffset += (actualCol * TRANS_BLOCK);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool NO_NEED_ALIGN=false>
    __aicore__ inline void Trans2(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        int64_t actualTotalUbCol = GetAlign(ubLoopInfo.totalUbCol, dstEleUbBlock_);
        if constexpr (NO_NEED_ALIGN) {
            actualTotalUbCol = ubLoopInfo.totalUbCol;
        }
        uint8_t repeatTimes = actualTotalUbCol * colRepeatNum_ / srcEleUbBlock_;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : 1;
        TransDataTo5HDParams transDataParams = {false, false, repeatTimes, dstRepStride, srcRepStride};
        uint64_t srcLocalList[TRANS_BLOCK];
        uint64_t dstLocalList[TRANS_BLOCK];
        if (sizeof(T1) == 2) {
            LocalTensor<half> srcLocalFP16 = srcLocal.template ReinterpretCast<half>();
            LocalTensor<half> dstLocalFP16 = dstLocal.template ReinterpretCast<half>();
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * TRANS_BLOCK;
                srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocalFP16[offset].GetPhyAddr());
            }
            for (int i = 0; i < TRANS_BLOCK; i++) {
                uint64_t offset = i * actualTotalUbCol * colRepeatNum_;
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocalFP16[offset].GetPhyAddr());
            }
            TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        } else {
            for (uint64_t i = 0; i < TRANS_BLOCK / srcEleUbBlock_; i++) {
                for (uint64_t j = 0; j < srcEleUbBlock_; j++) {
                    uint64_t offset = i * srcEleUbBlock_ + j * TRANS_BLOCK;
                    srcLocalList[i * srcEleUbBlock_ + j] =
                        reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                }
            }
            for (uint64_t i = 0; i < TRANS_BLOCK; i += 2) { // 2 is stride
                uint64_t offset = (i / 2)  * actualTotalUbCol * colRepeatNum_; // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            for (uint64_t i = 1; i < TRANS_BLOCK; i += 2) { // 2 is stride
                uint64_t offset = (i / 2 + srcEleUbBlock_) * actualTotalUbCol * colRepeatNum_;  // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            TransDataTo5HD<T1>(dstLocalList, srcLocalList, transDataParams);
        }
        
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UBRearrange4Concat(const UbLoopInfo& ubLoopInfo, LocalTensor<T1>& srcLocal, LocalTensor<T1>& dstLocal)
    {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        for (int64_t i = 0; i < ubLoopInfo.count; i++) {
            uint16_t blockCount = ubLoopInfo.currentUbRowFactor;
            uint16_t blockLen = ubLoopInfo.inputCol[i] / srcEleUbBlock_;
            uint16_t srcGap = 0;
            uint16_t dstGap = (ubLoopInfo.totalUbColAlign - ubLoopInfo.inputCol[i]) / srcEleUbBlock_;
            DataCopyParams copyParams{blockCount, blockLen, srcGap, dstGap};
            DataCopy(dstLocal[dstOffset], srcLocal[srcOffset], copyParams);
            srcOffset += blockCount * ubLoopInfo.inputCol[i];
            dstOffset += ubLoopInfo.inputCol[i];
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoCast(const UbLoopInfo& ubLoopInfo, uint32_t castCount)
    {
        if constexpr (sizeof(T1) == sizeof(T2)) {
            Cast(dstLocalFP32_, srcLocal_, RoundMode::CAST_NONE, castCount);
            Cast(srcLocalT2_, dstLocalFP32_, RoundMode::CAST_RINT, castCount);
            DataCopy(dstLocal_, srcLocalT2_, castCount);
        } else {
            Cast(dstLocal_, srcLocal_, RoundMode::CAST_NONE, castCount);
        }
    }

    __aicore__ inline void ComputeOneConcat(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(event_);
        WaitFlag<HardEvent::MTE2_V>(event_);
        if (ubLoopInfo.totalUbColAlign == 0) {
            return;
        }
        if constexpr (NEAD_CAST) {
            uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.totalUbColAlign;
            DoCast(ubLoopInfo, castCount);
        } else {
            DataCopy(dstLocalT1_, srcLocal_, ubLoopInfo.totalUbColAlign);
        }
    }

    __aicore__ inline void ComputeAllAlign(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(event_);
        WaitFlag<HardEvent::MTE2_V>(event_);
        UBRearrange4Concat(ubLoopInfo, srcLocal_, dstLocalT1_);
        if constexpr (NEAD_CAST) {
            DataCopy(srcLocal_, dstLocalT1_, ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor);
            PipeBarrier<PIPE_V>();
            uint32_t castCount = ubLoopInfo.currentUbRowFactor * ubLoopInfo.currentUbColFactor;
            DoCast(ubLoopInfo, castCount);
        }
    }

    __aicore__ inline void ComputeNotAlign(const UbLoopInfo& ubLoopInfo)
    {
        SetFlag<HardEvent::MTE2_V>(event_);
        WaitFlag<HardEvent::MTE2_V>(event_);
        // 3、ub重排
        UBRearrange4Trans(ubLoopInfo, srcLocal_, dstLocalT1_);
        // 4、跨block对齐转置
        Trans1(ubLoopInfo, dstLocalT1_, srcLocal_);
        // 5、ub重排
        UBRearrange4TransConcat(ubLoopInfo, srcLocal_, dstLocalT1_);
        // 6、跨block对齐转置
        Trans2(ubLoopInfo, dstLocalT1_, srcLocal_);

        // 7、cast or ubToub
        if constexpr (NEAD_CAST) {
            uint32_t castCount = ubLoopInfo.currentUbRowFactor * GetAlign(ubLoopInfo.currentUbColFactor, dstEleUbBlock_);
            DoCast(ubLoopInfo, castCount);
        } else {
            DataCopy(dstLocalT1_, srcLocal_, ubLoopInfo.currentUbRowFactor * GetAlign(ubLoopInfo.currentUbColFactor, dstEleUbBlock_));
        }
    }

private:
    bool isAllAlign_{false};
    bool isHalfAlign_{false};
    bool isOneConcat_{false};
    int64_t blockIdx_{0};
    int64_t inputNum_{0};
    int64_t ubRowFactor_{0};
    int64_t ubColFactor_{0};
    int64_t srcEleUbBlock_{0};
    int64_t dstEleUbBlock_{0};
    int64_t dim_{0};
    int64_t numChunk_{0};
    int64_t outputRow_{0};
    int64_t outputCol_{0};
    int64_t blockRowFactor_{0};
    int64_t blockColFactor_{0};
    int64_t tailBlockRowFactor_{0};
    int64_t tailBlockColFactor_{0};
    int64_t blockRowGroup_{0};
    int64_t blockColGroup_{0};
    int64_t currentBlockRowFactor_{0};
    int64_t currentBlockColFactor_{0};
    int64_t colRepeatNum_{0};

    TPipe *pipe_;
    TEventID event_{0};
    TensorDesc<T1> desc_;
    ListTensorDesc inputList_;
    GlobalTensor<T2> dstGlobal_;
    GlobalTensor<T1> srcGlobal_;
    TBuf<AscendC::TPosition::VECCALC> srcBuf_;
    TBuf<AscendC::TPosition::VECCALC> dstBuf_;
    LocalTensor<T1> srcLocal_;
    LocalTensor<T2> srcLocalT2_;
    LocalTensor<T2> dstLocal_;
    LocalTensor<T1> dstLocalT1_;
    LocalTensor<float> dstLocalFP32_;
};
#endif // _CHUNK_CAT_DATA_H_