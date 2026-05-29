# MulNoNan

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

- **算子功能**：二元 element-wise 安全乘法，把 `Mul` 中
  `0 · inf = NaN` 与 `0 · NaN = NaN` 两类异常屏蔽为 0，避免在反向传播
  / loss 计算等场景把 NaN 污染传递下去。
- **计算公式**：

$$
y_i = \begin{cases}
0,                       & x_{2,i} = 0 \\
x_{1,i} \cdot x_{2,i},   & x_{2,i} \ne 0
\end{cases}
$$

  仅判 `x2` 一侧。`x2 = -0` 同样进入零臂（IEEE-754 `-0 == 0`）。
  `x2 \ne 0` 时所有 IEEE 行为按普通乘法保留（NaN / Inf 正常传播）。
- **第三方对标**：等价于 TensorFlow `tf.math.multiply_no_nans`。

## 参数说明

| 参数名 | 输入/输出 | 描述                                                 | 数据类型                              | 数据格式 |
| :----: | :-------: | :--------------------------------------------------- | :-----------------------------------: | :------: |
| x1     | 输入      | 乘法第一个输入张量。                                 | FLOAT16, FLOAT, INT32, BFLOAT16       | ND       |
| x2     | 输入      | 乘法第二个输入张量；判 0 主体；与 `x1` 须可广播。    | FLOAT16, FLOAT, INT32, BFLOAT16       | ND       |
| y      | 输出      | `x1 * x2` 的结果，`x2 == 0` 处强制为 0。shape 为 `x1`、`x2` 广播后的统一形状。 | FLOAT16, FLOAT, INT32, BFLOAT16 | ND       |

## 约束说明

- `x1`、`x2`、`y` 必须为**同一种 dtype**；不支持 mix-dtype。
- 支持任意 NumPy 广播形态（含标量 `[1]`、单维 broadcast、跨 rank broadcast）。
- 支持动态 shape 与动态 rank。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/mul_no_nan_proto.h` | `REG_OP(MulNoNan)`，二输入一输出 |
| 算子定义 | `op_host/mul_no_nan_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/mul_no_nan_infershape.cpp` | 复用 `Ops::Base::InferShape4Broadcast` |
| Tiling | `op_host/arch35/mul_no_nan_tiling_arch35.{h,cpp}` | 按 dtype 分支调用 `Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/mul_no_nan_dag.h` | fp32/int32 通路原生计算；fp16/bf16 通路提升 fp32 中间精度 |
| Struct | `op_kernel/arch35/mul_no_nan_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel 入口 | `op_kernel/mul_no_nan_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32 / int32 通路

```
In0/In1 -- CopyInBrc --+
                       +-- Mul(x1,x2) --+
                                        +-- Select(mask, MulRes, Zero) -- CopyOut -- Out0
            Const(0)  -- Duplicate -- Zero --+
                                             |
                       +-- Compare(NE, x2, Zero) -- mask
```

`Vec::Compare<u8, T, NE>` 输出位掩码，`Vec::Select<u8, T, TENSOR_TENSOR>` 按
`mask ? MulRes : Zero` 逐元素选择。`x2 == 0` 时即使 `MulRes` 是 `NaN`（`0 · inf`
/ `0 · NaN`）也被 Select 丢弃，输出 0。

### fp16 / bf16 通路

```
In0/In1 -- CopyInBrc -- Cast(->fp32) --+
                                       +-- Mul --+
                                                 +-- Select -- Cast(->T,RINT) -- CopyOut -- Out0
            Const(0,fp32) -- Duplicate -- Zero --+
                                                 |
                              +-- Compare(NE) -- mask
```

把 fp16 / bf16 整体提升到 fp32 做 cmp/sel/mul，末端用 `CAST_MODE_RINT`
（round-to-nearest-even）回退。原因：
1. 与 DSL `mul_no_nan.py` 在 fp16 `vcmpsel` 不可用时 fallback 到 fp32 的行为一致；
2. 避免 fp16 中间溢出（如 `3e4 · 3e4` 在 fp16 中先 inf 再被 select 处理会
   引入额外的 saturation 不确定性），fp32 中间有充足动态范围。

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_mul_no_nan](examples/test_geir_mul_no_nan.cpp) | 通过[算子IR](op_graph/mul_no_nan_proto.h)构图方式调用 MulNoNan 算子；覆盖 fp32/fp16/bf16/int32 基础用例 + `0·inf`、`0·NaN`、`-0`、广播等关键特殊值用例。 |
