# FusedMulAddAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

> 本算子目前仅在 Ascend 950 (dav-3510 / Atlas 350 加速卡) 架构上实现。

## 功能说明

- **算子功能**：四元 element-wise 融合算子，把 `Mul → Add → Add` 子图融合为
  单次 kernel 启动，相对未融合实现减少两次 GM 中间数据搬运，常用于
  `BatchMatmul + bias + residual` 等模式。
- **计算公式**：

$$y = x_1 \cdot x_2 + x_3 + x_4$$

  四个输入按 NumPy 广播规则两两对齐后逐元素计算，计算顺序固定不可交换。

## 参数说明

| 参数名 | 输入/输出 | 描述                                                         | 数据类型                  | 数据格式 |
| :----: | :-------: | :----------------------------------------------------------- | :-----------------------: | :------: |
| x1     | 输入      | 乘法第一个输入张量。                                         | FLOAT16, FLOAT, INT32     | ND       |
| x2     | 输入      | 乘法第二个输入张量；与 `x1` 须可广播。                       | FLOAT16, FLOAT, INT32     | ND       |
| x3     | 输入      | 第一次加法的输入张量；与 `x1 * x2` 结果须可广播。            | FLOAT16, FLOAT, INT32     | ND       |
| x4     | 输入      | 第二次加法的输入张量；与 `x1 * x2 + x3` 结果须可广播。       | FLOAT16, FLOAT, INT32     | ND       |
| y      | 输出      | `x1 * x2 + x3 + x4` 的结果；shape 为四者广播后的统一形状。   | FLOAT16, FLOAT, INT32     | ND       |

## 约束说明

- `x1`、`x2`、`x3`、`x4`、`y` 必须为**同一种 dtype**；不支持 mix-dtype；不支持 bf16。
- 支持任意 NumPy 广播形态（含标量 `[1]`、单维 broadcast、跨 rank broadcast）。
- 支持动态 shape 与动态 rank。
- 计算顺序为 `((x1 * x2) + x3) + x4`，不可交换；对浮点输入会先 Cast 到 fp32 计算
  再 Cast 回输入 dtype，避免半精度累加误差。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/fused_mul_add_add_proto.h` | `REG_OP(FusedMulAddAdd)`，四输入一输出 |
| 算子定义 | `op_host/fused_mul_add_add_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/fused_mul_add_add_infershape.cpp` | 复用 `Ops::Base::InferShape4Broadcast(ctx, 4)` |
| Tiling | `op_host/arch35/fused_mul_add_add_tiling_arch35.{h,cpp}` | 按 dtype 分支调用 `Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/fused_mul_add_add_dag.h` | fp32/fp16 通路在 fp32 中间精度下用 `Vec::Mul + Vec::Add + Vec::Add`；int32 通路用 `Vec::Mul + Vec::Add + Vec::Add` |
| Struct | `op_kernel/arch35/fused_mul_add_add_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel 入口 | `op_kernel/fused_mul_add_add_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp16 / fp32 通路

```
In0/In1/In2/In3 -- CopyInBrc -- Cast(->fp32) -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Vec::Add(+x4) -- Cast(->T,RINT) -- CopyOut -- Out0
```

全部输入先 Cast 到 fp32，再用 `Vec::Mul + Vec::Add + Vec::Add` 三段式按公式
`y = x1 * x2 + x3 + x4` 顺序计算，最后 Cast 回 T 写出。

### int32 通路

```
In0/In1/In2/In3 -- CopyInBrc -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Vec::Add(+x4) -- CopyOut -- Out0
```
