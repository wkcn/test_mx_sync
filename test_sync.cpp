#include <cstddef>
#include <cstdint>
#include <iostream>
using namespace std;

#define DEFAULT(x)

struct Context {
  /*! \brief Type of device */
  enum DeviceType {
    kCPU = 1 << 0,
    kGPU = 1 << 1,
    kCPUPinned = 3,
    kCPUShared = 5,
  };
  /*! \brief the device type we run the op on */
  DeviceType dev_type;
  /*! \brief device id we are going to run it on */
  int32_t dev_id;
};

/*!
 * \brief execution time context.
 *  The information needed in runtime for actual execution.
 */
struct RunContext {
  /*! \brief base Context */
  Context ctx;
  /*!
   * \brief the stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *stream;
  /*!
   * \brief the auxiliary stream of the device, can be NULL or Stream<gpu>* in GPU mode
   */
  void *aux_stream;
  /*!
   * \brief indicator of whether this execution is run in bulk mode
   */
  bool is_bulk;
  void* get_gpu_stream() {
    // the first element is stream
    uintptr_t *p = static_cast<uintptr_t*>(stream); 
    return reinterpret_cast<void*>(*p);
  }
};

/*! \brief Engine asynchronous operation */
typedef void (*EngineAsyncFunc)(void*, void*, void*);
/*! \brief Engine synchronous operation */
typedef void (*EngineSyncFunc)(void*, void*);
/*! \brief Callback to free the param for EngineAsyncFunc/EngineSyncFunc */
typedef void (*EngineFuncParamDeleter)(void*);
/*! \brief handle to NDArray */
typedef void *NDArrayHandle;
/*! \brief handle to Engine FnProperty */
typedef const void *EngineFnPropertyHandle;
/*! \brief handle to Context */
typedef const void *ContextHandle;

/*! \brief Function property, used to hint what action is pushed to engine. */
enum class FnProperty {
  /*! \brief Normal operation */
  kNormal,
  /*! \brief Copy operation from GPU to other devices */
  kCopyFromGPU,
  /*! \brief Copy operation from CPU to other devices */
  kCopyToGPU,
  /*! \brief Prioritized sync operation on CPU */
  kCPUPrioritized,
  /*! \brief Asynchronous function call */
  kAsync,
  /*! \brief Delete variable call */
  kDeleteVar,
  /*! \brief Prioritized sync operation on GPU */
  kGPUPrioritized,
  /*! \brief Operation not to be skipped even with associated exception */
  kNoSkip
};  // enum class FnProperty

/*!
 * \brief OnComplete Callback to the engine,
 *  called by AsyncFn when action completes
 */
class CallbackOnComplete {
 public:
  // use implicit copy and assign
  /*! \brief involve the callback */
  inline void operator()(const void* error = nullptr) const {
    (*callback_)(engine_, param_, error);
  }

 private:
  /*! \brief the real callback */
  void (*callback_)(void *, void *, const void *);
  /*! \brief the engine class passed to callback */
  void* engine_;
  /*! \brief the parameter set on callback */
  void* param_;
};

int (*MXEnginePushAsyncND)(EngineAsyncFunc async_func, void* func_param,
                                EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                                NDArrayHandle const_nds_handle, int num_const_nds,
                                NDArrayHandle mutable_nds_handle, int num_mutable_nds,
                                EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                                int priority DEFAULT(0), const char* opr_name DEFAULT(NULL),
                                bool wait DEFAULT(false));

int (*MXEnginePushSyncND)(EngineSyncFunc sync_func, void* func_param,
                               EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                               NDArrayHandle const_nds_handle, int num_const_nds,
                               NDArrayHandle mutable_nds_handle, int num_mutable_nds,
                               EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                               int priority DEFAULT(0), const char* opr_name DEFAULT(NULL));

struct ParamStruct {
  void* data;
  int N;
};

void sync_func_inst(void* rctx, void* param) {
  ParamStruct *ps = static_cast<ParamStruct*>(param);
  float* data = static_cast<float*>(ps->data);
  for (int i = 0; i < ps->N; ++i) {
    data[i] += 1;
  }
}

void async_func_inst(void* rctx, void* on_complete, void* param) {
  ParamStruct *ps = static_cast<ParamStruct*>(param);
  float* data = static_cast<float*>(ps->data);
  for (int i = 0; i < ps->N; ++i) {
    data[i] += 1;
  }
  (*static_cast<CallbackOnComplete*>(on_complete))();
}

void deleter_inst(void* param) {
  ParamStruct *ps = static_cast<ParamStruct*>(param);
  delete ps;
}

extern "C" {
  void SetMXEnginePushSyncND(void* func) {
    MXEnginePushSyncND = reinterpret_cast<decltype(MXEnginePushSyncND)>(func); 
  }
  void SetMXEnginePushAsyncND(void* func) {
    MXEnginePushAsyncND = reinterpret_cast<decltype(MXEnginePushAsyncND)>(func); 
  }
  void AddOne(void* ndarray, void* data, const int N) {
    ParamStruct *ps = new ParamStruct{data, N};
    Context ctx{Context::kCPU, 0};
    MXEnginePushSyncND(sync_func_inst, static_cast<void*>(ps), deleter_inst, &ctx, nullptr, 0,
      ndarray, 1, nullptr, 0, nullptr); 
  }
  void AddOneAsync(void* ndarray, void* data, const int N) {
    ParamStruct *ps = new ParamStruct{data, N};
    Context ctx{Context::kCPU, 0};
    MXEnginePushAsyncND(async_func_inst, static_cast<void*>(ps), deleter_inst, &ctx, nullptr, 0,
      ndarray, 1, nullptr, 0, nullptr, false); 
  }
};
