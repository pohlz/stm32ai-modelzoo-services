/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-09-02T16:01:10+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x2b3e13d7642bf5a8ed53b46ddcce5f06"
#define _STAI_NETWORK_DATETIME            "2025-09-02T16:01:10+0200"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        (STAI_NETWORK_CONTEXT_ALIGNMENT)

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(11, 1, 0),
  .tool_version = STAI_INIT_VERSION(3, 1, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 3, 32),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 7),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 16768),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 194524),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0025988586712628603f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0025988586712628603f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0032046164851635695f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09998618066310883f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(model_batch_normalization_batchnorm_mul_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.41270145773887634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10868505388498306f),
    AI_PACK_INTQ_ZP(-108)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(model_batch_normalization_batchnorm_sub_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.042500630021095276f),
    AI_PACK_INTQ_ZP(124)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5240471363067627f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01964247040450573f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(model_batch_normalization_1_batchnorm_mul_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003952982369810343f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021467437967658043f),
    AI_PACK_INTQ_ZP(-106)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(model_batch_normalization_1_batchnorm_sub_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019602524116635323f),
    AI_PACK_INTQ_ZP(116)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  transpose_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2944, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2944, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  model_batch_normalization_batchnorm_mul_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2944, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  model_batch_normalization_batchnorm_sub_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  model_batch_normalization_1_batchnorm_mul_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1792, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  model_batch_normalization_1_batchnorm_sub_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 3), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  transpose_1_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 1, 32), AI_STRIDE_INIT(4, 1, 1, 3, 3),
  1, &transpose_1_output_array, &transpose_1_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 23, 1), AI_STRIDE_INIT(4, 1, 1, 128, 2944),
  1, &conv2d_3_output_array, &conv2d_3_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_4_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 23, 1), AI_STRIDE_INIT(4, 1, 1, 128, 2944),
  1, &eltwise_4_output_array, &eltwise_4_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  model_batch_normalization_batchnorm_mul_4D, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &model_batch_normalization_batchnorm_mul_4D_array, &model_batch_normalization_batchnorm_mul_4D_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_5_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 23, 1), AI_STRIDE_INIT(4, 1, 1, 128, 2944),
  1, &eltwise_5_output_array, &eltwise_5_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  model_batch_normalization_batchnorm_sub_4D, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &model_batch_normalization_batchnorm_sub_4D_array, &model_batch_normalization_batchnorm_sub_4D_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 14, 1), AI_STRIDE_INIT(4, 1, 1, 128, 1792),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_9_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 14, 1), AI_STRIDE_INIT(4, 1, 1, 128, 1792),
  1, &eltwise_9_output_array, &eltwise_9_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  model_batch_normalization_1_batchnorm_mul_4D, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &model_batch_normalization_1_batchnorm_mul_4D_array, &model_batch_normalization_1_batchnorm_mul_4D_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 14, 1), AI_STRIDE_INIT(4, 1, 1, 128, 1792),
  1, &eltwise_10_output_array, &eltwise_10_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  model_batch_normalization_1_batchnorm_sub_4D, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &model_batch_normalization_1_batchnorm_sub_4D_array, &model_batch_normalization_1_batchnorm_sub_4D_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_1_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_1_chain,
  NULL, &transpose_1_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_output, &model_batch_normalization_batchnorm_mul_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_4_layer, 4,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_4_chain,
  NULL, &eltwise_4_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_4_output, &model_batch_normalization_batchnorm_sub_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_5_layer, 5,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_5_chain,
  NULL, &eltwise_5_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_8_output, &model_batch_normalization_1_batchnorm_mul_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_9_layer, 9,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_9_chain,
  NULL, &eltwise_9_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_9_output, &model_batch_normalization_1_batchnorm_sub_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_10_chain,
  NULL, &eltwise_10_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_transpose_1(_stai_network_context* net_ctx)
{
  conversion_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2848);
  conversion_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2848);
  transpose_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 2944);
  transpose_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2944);
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, { conversion_0_output.data->data});
  forward_transpose(&transpose_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, { transpose_1_output.data->data});
}
void forward_lite_eltwise_4(_stai_network_context* net_ctx)
{
  conv2d_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  model_batch_normalization_batchnorm_mul_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  model_batch_normalization_batchnorm_mul_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  eltwise_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 2944);
  eltwise_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2944);
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 2, { conv2d_3_output.data->data,model_batch_normalization_batchnorm_mul_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_4_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, { eltwise_4_output.data->data});
}
void forward_lite_eltwise_5(_stai_network_context* net_ctx)
{
  eltwise_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 2944);
  eltwise_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2944);
  model_batch_normalization_batchnorm_sub_4D_array.data = AI_PTR(net_ctx->_weights[0] + 128);
  model_batch_normalization_batchnorm_sub_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 128);
  eltwise_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 2, { eltwise_4_output.data->data,model_batch_normalization_batchnorm_sub_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, { eltwise_5_output.data->data});
}
void forward_lite_eltwise_9(_stai_network_context* net_ctx)
{
  conv2d_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 14976);
  conv2d_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14976);
  model_batch_normalization_1_batchnorm_mul_4D_array.data = AI_PTR(net_ctx->_weights[0] + 256);
  model_batch_normalization_1_batchnorm_mul_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 256);
  eltwise_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 2, { conv2d_8_output.data->data,model_batch_normalization_1_batchnorm_mul_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, { eltwise_9_output.data->data});
}
void forward_lite_eltwise_10(_stai_network_context* net_ctx)
{
  eltwise_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  model_batch_normalization_1_batchnorm_sub_4D_array.data = AI_PTR(net_ctx->_weights[0] + 384);
  model_batch_normalization_1_batchnorm_sub_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 384);
  eltwise_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 1792);
  eltwise_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1792);
  _STAI_NETWORK_EVENT_NODE_START_CB(10, 2, { eltwise_9_output.data->data,model_batch_normalization_1_batchnorm_sub_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(10, 1, { eltwise_10_output.data->data});
}

/*****************************************************************************/


STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conversion_0 */
  {
      const ai_float* t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_inputs[0] + 0);
    ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 2848);
    const ai_u32 t_out_0_shape_h_w_ch_d_prod_const_u32 = 96;
    const ai_float t_out_0_fmt_scale_const_f32 = 0.0025988586712628603f;
    const ai_i8 t_out_0_fmt_zero_const_s8 = -128;
  
  _STAI_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(t_in_0_ptr_const_f32, t_out_0_ptr_s8, t_out_0_shape_h_w_ch_d_prod_const_u32, t_out_0_fmt_scale_const_f32, t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conversion_0 */
  /* LITE_KERNEL_SECTION BEGIN transpose_1 */
  {
    
  forward_lite_transpose_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END transpose_1 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_3 */
  {
      const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2944);
    const ai_u16 t_in_0_shape_w_const_u16 = 32;
    const ai_u16 t_in_0_shape_h_const_u16 = 1;
    const ai_u16 t_in_0_shape_ch_const_u16 = 3;
    const ai_i8* t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 512);
    const ai_u16 t_out_0_shape_ch_const_u16 = 128;
    const ai_u16 t_weight_0_shape_w_const_u16 = 10;
    const ai_u16 t_weight_0_shape_h_const_u16 = 1;
    const ai_u16 l_stride_1_const_u16 = 1;
    const ai_u16 l_stride_0_const_u16 = 1;
    const ai_i32 l_pad_W_0_const_s32 = 0;
    const ai_i32 l_pad_H_0_const_s32 = 0;
    const ai_i32* t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4352);
    const ai_i8 t_in_0_fmt_zero_const_s8 = -128;
    const ai_i8 t_out_0_fmt_zero_const_s8 = -128;
    const ai_float t_in_0_fmt_scale_const_f32 = 0.0025988586712628603f;
    const ai_float t_out_0_fmt_scale_const_f32 = 0.0032046164851635695f;
    const ai_float t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009083469398319721f, 0.008578579872846603f, 0.010237965732812881f, 0.005520873703062534f, 0.005494991783052683f, 0.00891890749335289f, 0.011159857735037804f, 0.0006068773800507188f, 0.013184693641960621f, 0.025127455592155457f, 0.0012274719774723053f, 0.009503321722149849f, 0.012235403060913086f, 0.01784079521894455f, 0.010673223994672298f, 0.007991847582161427f, 0.008287186734378338f, 0.011806926690042019f, 0.0008755203452892601f, 0.01246015727519989f, 0.011096368543803692f, 0.015258044935762882f, 0.016014650464057922f, 0.01030734647065401f, 0.017796657979488373f, 0.012248688377439976f, 0.0007699226262047887f, 0.03332092612981796f, 0.007701249793171883f, 0.01801934652030468f, 0.007961710914969444f, 0.010041971690952778f, 0.005922364536672831f, 0.011058688163757324f, 0.0006132502458058298f, 0.0005872449255548418f, 0.0017316839657723904f, 0.0096902372315526f, 0.011693034321069717f, 0.013360156677663326f, 0.0016353466780856252f, 0.012522482313215733f, 0.015409175306558609f, 0.006280376575887203f, 0.008431644178926945f, 0.0005599536234512925f, 0.0006980589823797345f, 0.003007000545039773f, 0.015416564419865608f, 0.012589899823069572f, 0.0004892538418062031f, 0.012277690693736076f, 0.007081584073603153f, 0.0006505729979835451f, 0.01390975434333086f, 0.006817939225584269f, 0.01353229209780693f, 0.007940913550555706f, 0.0005906260339543223f, 0.001022922690026462f, 0.008589851669967175f, 0.010120589286088943f, 0.001709073898382485f, 0.010506669990718365f, 0.0014032174367457628f, 0.0006913129473105073f, 0.011486840434372425f, 0.01198412012308836f, 0.01146608218550682f, 0.011251379735767841f, 0.0008923689601942897f, 0.025158917531371117f, 0.01969641074538231f, 0.0005822685780003667f, 0.010658581741154194f, 0.001157995779067278f, 0.016355177387595177f, 0.014233953319489956f, 0.010597770102322102f, 0.0077719795517623425f, 0.002319705206900835f, 0.00897210743278265f, 0.011329783126711845f, 0.0009281691745854914f, 0.0029155854135751724f, 0.009470868855714798f, 0.0006398052209988236f, 0.010699199512600899f, 0.02092788554728031f, 0.012611250393092632f, 0.014832488261163235f, 0.03707197681069374f, 0.011088087223470211f, 0.01368359662592411f, 0.008701751939952374f, 0.01054022554308176f, 0.008418289944529533f, 0.01667618378996849f, 0.0007055875612422824f, 0.012992197647690773f, 0.011246822774410248f, 0.000954869668930769f, 0.013939943164587021f, 0.01246354915201664f, 0.007268659304827452f, 0.010397069156169891f, 0.006714244838804007f, 0.0076147643849253654f, 0.008435252122581005f, 0.009274451993405819f, 0.0009557245648466051f, 0.008913584984838963f, 0.007876447401940823f, 0.00053571374155581f, 0.0008207489736378193f, 0.006910750642418861f, 0.014628642238676548f, 0.03565946966409683f, 0.017915816977620125f, 0.0010467586107552052f, 0.0018274557078257203f, 0.0005944138392806053f, 0.000601522158831358f, 0.019044384360313416f, 0.012665711343288422f, 0.0015248791314661503f, 0.004578185733407736f, 0.01349453441798687f);
    const ai_layer_format_type l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
    ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_u16 t_out_0_shape_w_const_u16 = 23;
    const ai_u16 t_out_0_shape_h_const_u16 = 1;
    ai_i16* t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3040);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(t_in_0_ptr_const_s8, t_in_0_shape_w_const_u16, t_in_0_shape_h_const_u16, t_in_0_shape_ch_const_u16, t_weight_0_ptr_const_s8, t_out_0_shape_ch_const_u16, t_weight_0_shape_w_const_u16, t_weight_0_shape_h_const_u16, l_stride_1_const_u16, l_stride_0_const_u16, l_pad_W_0_const_s32, l_pad_H_0_const_s32, t_weight_1_ptr_const_s32, t_in_0_fmt_zero_const_s8, t_out_0_fmt_zero_const_s8, t_in_0_fmt_scale_const_f32, t_out_0_fmt_scale_const_f32, t_weight_0_fmt_scale_const_f32, l_out_ch_format_const_layer_format_type, t_out_0_ptr_s8, t_out_0_shape_w_const_u16, t_out_0_shape_h_const_u16, 1, 7032, t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_3 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_4 */
  {
    
  forward_lite_eltwise_4(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_4 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_5 */
  {
    
  forward_lite_eltwise_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_5 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_8 */
  {
      const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_u16 t_in_0_shape_w_const_u16 = 23;
    const ai_u16 t_in_0_shape_h_const_u16 = 1;
    const ai_u16 t_in_0_shape_ch_const_u16 = 128;
    const ai_i8* t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4864);
    const ai_u16 t_out_0_shape_ch_const_u16 = 128;
    const ai_u16 t_weight_0_shape_w_const_u16 = 10;
    const ai_u16 t_weight_0_shape_h_const_u16 = 1;
    const ai_u16 l_stride_1_const_u16 = 1;
    const ai_u16 l_stride_0_const_u16 = 1;
    const ai_i32* t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 168704);
    const ai_i8 t_in_0_fmt_zero_const_s8 = -108;
    const ai_i8 t_out_0_fmt_zero_const_s8 = -128;
    const ai_float t_in_0_fmt_scale_const_f32 = 0.10868505388498306f;
    const ai_float t_out_0_fmt_scale_const_f32 = 0.5240471363067627f;
    const ai_float t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.013552902266383171f, 0.01056675985455513f, 0.01759556122124195f, 0.017382066696882248f, 0.013553116470575333f, 0.018288234248757362f, 0.012505964376032352f, 0.016083599999547005f, 0.03536554425954819f, 0.01244861539453268f, 0.010411672294139862f, 0.01529436931014061f, 0.014650939963757992f, 0.011721662245690823f, 0.018667763099074364f, 0.013781283982098103f, 0.015559643507003784f, 0.015227960422635078f, 0.013336687348783016f, 0.012480532750487328f, 0.011538411490619183f, 0.013580349273979664f, 0.014194871298968792f, 0.011822710745036602f, 0.021499160677194595f, 0.012267236597836018f, 0.013508752919733524f, 0.011835966259241104f, 0.015164628624916077f, 0.01386450044810772f, 0.019320880994200706f, 0.015536430291831493f, 0.017316734418272972f, 0.011294466443359852f, 0.012568130157887936f, 0.010512377135455608f, 0.0138936722651124f, 0.011662456206977367f, 0.015513673424720764f, 0.012601878494024277f, 0.01795579493045807f, 0.019215833395719528f, 0.013608031906187534f, 0.017363864928483963f, 0.014453349635004997f, 0.017249545082449913f, 0.012247244827449322f, 0.013989427126944065f, 0.01415931899100542f, 0.016958389431238174f, 0.021602865308523178f, 0.010652784258127213f, 0.012242649681866169f, 0.012326154857873917f, 0.01405242644250393f, 0.015881899744272232f, 0.012827517464756966f, 0.016819113865494728f, 0.01600673794746399f, 0.014821458607912064f, 0.014386026188731194f, 0.017839238047599792f, 0.013467943295836449f, 0.014467303641140461f, 0.010459478944540024f, 0.013042601756751537f, 0.01172505784779787f, 0.010694592259824276f, 0.016798779368400574f, 0.01254766620695591f, 0.019053827971220016f, 0.016089359298348427f, 0.021839920431375504f, 0.013031262904405594f, 0.013200921006500721f, 0.013176246546208858f, 0.013637319207191467f, 0.015933139249682426f, 0.0130311856046319f, 0.01302524097263813f, 0.010621175169944763f, 0.011723176576197147f, 0.010707876645028591f, 0.013115460984408855f, 0.014023978263139725f, 0.013297457247972488f, 0.012704744935035706f, 0.017539147287607193f, 0.014530389569699764f, 0.01211971789598465f, 0.012134169228374958f, 0.014397574588656425f, 0.015349606052041054f, 0.01566011644899845f, 0.012138372287154198f, 0.016908856108784676f, 0.019541006535291672f, 0.010786944068968296f, 0.013714632019400597f, 0.01171032339334488f, 0.010595444589853287f, 0.013496856205165386f, 0.012129625305533409f, 0.010916950181126595f, 0.011200327426195145f, 0.0119522949680686f, 0.011695783585309982f, 0.019698766991496086f, 0.013923639431595802f, 0.016077542677521706f, 0.012275476939976215f, 0.01568695902824402f, 0.014417139813303947f, 0.014326225034892559f, 0.014655211940407753f, 0.012720770202577114f, 0.014101153239607811f, 0.014706308022141457f, 0.013269878923892975f, 0.013599921017885208f, 0.015037955716252327f, 0.012931951321661472f, 0.011151023209095001f, 0.017091067507863045f, 0.012693887576460838f, 0.01872381381690502f, 0.013840558007359505f, 0.01436083298176527f);
    const ai_layer_format_type l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
    ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14976);
    const ai_u16 t_out_0_shape_w_const_u16 = 14;
    const ai_u16 t_out_0_shape_h_const_u16 = 1;
    ai_i16* t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2944);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(t_in_0_ptr_const_s8, t_in_0_shape_w_const_u16, t_in_0_shape_h_const_u16, t_in_0_shape_ch_const_u16, t_weight_0_ptr_const_s8, t_out_0_shape_ch_const_u16, t_weight_0_shape_w_const_u16, t_weight_0_shape_h_const_u16, l_stride_1_const_u16, l_stride_0_const_u16, t_weight_1_ptr_const_s32, t_in_0_fmt_zero_const_s8, t_out_0_fmt_zero_const_s8, t_in_0_fmt_scale_const_f32, t_out_0_fmt_scale_const_f32, t_weight_0_fmt_scale_const_f32, l_out_ch_format_const_layer_format_type, t_out_0_ptr_s8, t_out_0_shape_w_const_u16, t_out_0_shape_h_const_u16, 1, 1, 12032, t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_8 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_9 */
  {
    
  forward_lite_eltwise_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_9 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_10 */
  {
    
  forward_lite_eltwise_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_10 */
  /* LITE_KERNEL_SECTION BEGIN pool_13 */
  {
      const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1792);
    ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_u16 t_in_0_shape_w_const_u16 = 1;
    const ai_u16 t_in_0_shape_h_const_u16 = 14;
    const ai_u16 t_in_0_shape_ch_const_u16 = 128;
    const ai_u16 l_pool_size_1_const_u16 = 1;
    const ai_u16 l_pool_size_0_const_u16 = 4;
    const ai_u16 l_legacy_pool_pad_1_const_u16 = 0;
    const ai_u16 l_legacy_pool_pad_0_const_u16 = 0;
    const ai_u16 l_pool_stride_1_const_u16 = 1;
    const ai_u16 l_pool_stride_0_const_u16 = 4;
    const ai_u16 t_out_0_shape_w_const_u16 = 1;
    const ai_u16 t_out_0_shape_h_const_u16 = 3;
    const ai_i8 t_in_0_fmt_zero_const_s8 = -106;
    const ai_i8 t_out_0_fmt_zero_const_s8 = -106;
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_maxpool_is8os8_scalepos(t_in_0_ptr_const_s8, t_out_0_ptr_s8, t_in_0_shape_w_const_u16, t_in_0_shape_h_const_u16, t_in_0_shape_ch_const_u16, l_pool_size_1_const_u16, l_pool_size_0_const_u16, l_legacy_pool_pad_1_const_u16, l_legacy_pool_pad_0_const_u16, l_pool_stride_1_const_u16, l_pool_stride_0_const_u16, t_out_0_shape_w_const_u16, t_out_0_shape_h_const_u16, 1.0f, t_in_0_fmt_zero_const_s8, t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END pool_13 */
  /* LITE_KERNEL_SECTION BEGIN gemm_15 */
  {
      ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 169216);
    const ai_i32* t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 193792);
    const ai_i8 t_in_0_fmt_zero_const_s8 = -106;
    const ai_i8 t_out_0_fmt_zero_const_s8 = -128;
    const ai_u16 t_in_0_shape_ch_const_u16 = 384;
    const ai_u16 t_out_0_shape_ch_const_u16 = 64;
    const ai_u32 t_out_0_shape_h_w_prod_const_u32 = 1;
    const ai_float t_in_0_fmt_scale_const_f32 = 0.021467437967658043f;
    const ai_float t_out_0_fmt_scale_const_f32 = 0.049500931054353714f;
    const ai_float t_weight_0_fmt_scale_const_f32 = 0.014752916060388088f;
    ai_i16* t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(t_out_0_ptr_s8, t_in_0_ptr_const_s8, t_weight_0_ptr_const_s8, t_weight_1_ptr_const_s32, t_in_0_fmt_zero_const_s8, t_out_0_fmt_zero_const_s8, t_in_0_shape_ch_const_u16, t_out_0_shape_ch_const_u16, t_out_0_shape_h_w_prod_const_u32, t_in_0_fmt_scale_const_f32, t_out_0_fmt_scale_const_f32, t_weight_0_fmt_scale_const_f32, t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END gemm_15 */
  /* LITE_KERNEL_SECTION BEGIN gemm_16 */
  {
      ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
    const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 194048);
    const ai_i32* t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 194496);
    const ai_i8 t_in_0_fmt_zero_const_s8 = -128;
    const ai_i8 t_out_0_fmt_zero_const_s8 = 48;
    const ai_u16 t_in_0_shape_ch_const_u16 = 64;
    const ai_u16 t_out_0_shape_ch_const_u16 = 7;
    const ai_u32 t_out_0_shape_h_w_prod_const_u32 = 1;
    const ai_float t_in_0_fmt_scale_const_f32 = 0.049500931054353714f;
    const ai_float t_out_0_fmt_scale_const_f32 = 0.3440615236759186f;
    const ai_float t_weight_0_fmt_scale_const_f32 = 0.01925067789852619f;
    ai_i16* t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(t_out_0_ptr_s8, t_in_0_ptr_const_s8, t_weight_0_ptr_const_s8, t_weight_1_ptr_const_s32, t_in_0_fmt_zero_const_s8, t_out_0_fmt_zero_const_s8, t_in_0_shape_ch_const_u16, t_out_0_shape_ch_const_u16, t_out_0_shape_h_w_prod_const_u32, t_in_0_fmt_scale_const_f32, t_out_0_fmt_scale_const_f32, t_weight_0_fmt_scale_const_f32, t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END gemm_16 */
  /* LITE_KERNEL_SECTION BEGIN nl_17 */
  {
      ai_i8* t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 128);
    const ai_u32 t_in_0_shape_ch_prod_const_u32 = 7;
    ai_i32* t_scratch_0_ptr_s32 = (ai_i32*)(net_ctx->_activations[0] + 136);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_nl_softmax_is8os8(t_out_0_ptr_s8, t_in_0_ptr_const_s8, t_in_0_shape_ch_prod_const_u32, 1, 7, 1477732992, 25, -62, t_scratch_0_ptr_s32);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, {(stai_ptr) t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END nl_17 */
  /* LITE_KERNEL_SECTION BEGIN conversion_18 */
  {
      const ai_i8* t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_float* t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[0] + 0);
    const ai_u32 t_out_0_shape_h_w_ch_d_prod_const_u32 = 7;
    const ai_float t_in_0_fmt_scale_const_f32 = 0.00390625f;
    const ai_i8 t_in_0_fmt_zero_const_s8 = -128;
  
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, {(stai_ptr) t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(t_in_0_ptr_const_s8, t_out_0_ptr_f32, t_out_0_shape_h_w_ch_d_prod_const_u32, t_in_0_fmt_scale_const_f32, t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, {(stai_ptr) t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conversion_18 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 2848;

  net_ctx->_outputs[0] = activations[0] + 8;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

