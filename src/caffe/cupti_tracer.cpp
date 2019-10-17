#include <cstdio>
#include <vector>
#include <stdlib.h>
#include <memory>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <cupti.h>
#include "caffe/cupti_tracer.hpp"

#define CUPTI_CALL(call)                                                         \
    do {                                                                         \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS) {                                          \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(-1);                                                            \
        }                                                                        \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

namespace cupti_tracer {


void CUptiManager::InternalBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    //fprintf(stderr, "InternalBufferRequested\n");
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUptiManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = nullptr;
    //fprintf(stderr, "InternalBufferCompleted\n");
    if (validSize > 0) {
        do {
            auto status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                tracer_->ActivityCallback(*record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);
    }
    free(buffer);
}

void CUptiManager::Initialize(Tracer *tracer) {
    tracer_ = tracer;
    cuptiActivityRegisterCallbacks((CUpti_BuffersCallbackRequestFunc)BufferRequested,
                                   (CUpti_BuffersCallbackCompleteFunc)BufferCompleted);
}

CUptiManager *CUptiManager::Get() {
    static auto manager = new CUptiManager();
    //fprintf(stderr, "%p\n", manager);
    return manager;
}

Tracer::Tracer() {
    manager_ = CUptiManager::Get();
    manager_->Initialize(this);
}

void Tracer::start() {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
    //CUPTI_CALL(cuptiActivityEnable());

    CUPTI_CALL(cuptiSubscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(APICallback), this));
    CUPTI_CALL(cuptiEnableAllDomains(1, subscriber_));
    //CUPTI_CALL(cuptiEnableCallback(1, subscriber_,
    //                               CUPTI_CB_DOMAIN_RUNTIME_API,
    //                               CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020));
}

void Tracer::stop() {
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
    //CUPTI_CALL(cuptiActivityDisable());

    CUPTI_CALL(cuptiUnsubscribe(subscriber_));

    //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
    //CUPTI_CALL(cuptiEnableCallback(/*disable=*/1, subscriber_,
    //                               CUPTI_CB_DOMAIN_DRIVER_API,
    //                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(cuptiActivityFlushAll(0));
}

void Tracer::APICallback(void *userdata, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid, const void *cbdata) {
    //if (domain == CUPTI_CB_DOMAIN_RUNTIME_API and cbid == CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020)
    //    fprintf(stderr, "ConfigureCall\n\n");
    return;
    Tracer *tracer = reinterpret_cast<Tracer *>(userdata);
    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API: {
            auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
            fprintf(stderr, "RUNTIME_API cbid=%llu functionName=%s symbolName=%s\n\n", (unsigned long long)cbid, cbInfo->functionName, cbInfo->symbolName);
            //std::cout << cbid << std::endl;
            //std::cout << cbInfo->correlationId << std::endl;
            //std::cout << cbInfo->functionName << std::endl;
            //std::cout << cbInfo->symbolName << std::endl;
            break;
        }
        case CUPTI_CB_DOMAIN_DRIVER_API: {
            auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
            fprintf(stderr, "DRIVER_API cbid=%llu functionName=%s symbolName=%s\n\n", (unsigned long long)cbid, cbInfo->functionName, cbInfo->symbolName);
            break;
        }
        case CUPTI_CB_DOMAIN_RESOURCE: {
            //auto *cbInfo = reinterpret_cast<const CUpti_ResourceData *>(cbdata);
            fprintf(stderr, "RESOURCE cbid=%llu\n\n", (unsigned long long)cbid);
            break;
        }
        case CUPTI_CB_DOMAIN_SYNCHRONIZE: {
            //auto *cbInfo = reinterpret_cast<const CUpti_SynchronizeData *>(cbdata);
            fprintf(stderr, "SYNCHRONIZE\n\n");
            break;
        }
        case CUPTI_CB_DOMAIN_NVTX: {
            auto *cbInfo = reinterpret_cast<const CUpti_NvtxData *>(cbdata);
            fprintf(stderr, "NVTX functionName=%s\n\n",cbInfo->functionName);
            break;
        }
    }
}

void Tracer::ActivityCallback(const CUpti_Activity &record) {
    //int pid = (int)getpid();
    std::string logname = "cupti_trace.txt";
    FILE *log = fopen(logname.c_str(), "a");
    switch (record.kind) {
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
            //records_.push_back(Record{"MEMCPY", "",
            //                   (unsigned long long) (memcpy->start),
            //                   (unsigned long long) (memcpy->end),
            //                   memcpy->deviceId, memcpy->streamId, memcpy->correlationId});
            fprintf(log, "MEMCPY copyKind=%u [ %llu - %llu ] device=%u context=%u stream=%u correlation=%u runtime_correlation=%u duration=%llu bytes=%llu\n\n",
                (unsigned int) (memcpy->copyKind),
                (unsigned long long) (memcpy->start),
                (unsigned long long) (memcpy->end),
                memcpy->deviceId, memcpy->contextId, memcpy->streamId,
                memcpy->correlationId, memcpy->runtimeCorrelationId,
                (unsigned long long) (memcpy->end - memcpy->start),
                (unsigned long long) (memcpy->bytes));
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto *memset = reinterpret_cast<const CUpti_ActivityMemset *>(&record);
            //records_.push_back(Record{"MEMSET", "",
            //                   (unsigned long long) (memset->start),
            //                   (unsigned long long) (memset->end),
            //                   memset->deviceId, memset->streamId, memset->correlationId});
             fprintf(log, "MEMSET memoryKind=%u [ %llu - %llu ] device=%u context=%u stream=%u correlation=%u duration=%llu bytes=%llu\n\n",
                (unsigned int) (memset->memoryKind),
                (unsigned long long) (memset->start),
                (unsigned long long) (memset->end),
                memset->deviceId, memset->contextId, memset->streamId, memset->correlationId,
                (unsigned long long) (memset->end - memset->start),
                (unsigned long long) (memset->bytes));
            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(&record);
            //records_.push_back(Record{"DRIVER", "",
            //                   (unsigned long long) (api->start),
            //                   (unsigned long long) (api->end),
            //                   api->processId, api->threadId, api->correlationId});
            fprintf(log, "DRIVER cbid=%u [ %llu - %llu ] process=%u thread=%u correlation=%u duration=%llu\n\n",
                api->cbid,
                (unsigned long long) (api->start),
                (unsigned long long) (api->end),
                api->processId, api->threadId, api->correlationId,
                (unsigned long long) (api->end - api->start));
            break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(&record);
            //records_.push_back(Record{"RUNTIME", "",
            //                   (unsigned long long) (api->start),
            //                   (unsigned long long) (api->end),
            //                   api->processId, api->threadId, api->correlationId});
            fprintf(log, "RUNTIME cbid=%u [ %llu - %llu ] process=%u thread=%u correlation=%u duration=%llu\n\n",
                api->cbid,
                (unsigned long long) (api->start),
                (unsigned long long) (api->end),
                api->processId, api->threadId, api->correlationId,
                (unsigned long long) (api->end - api->start));
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto *kernel = reinterpret_cast<const CUpti_ActivityKernel4 *>(&record);
            const char* kindString = (record.kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
            //records_.push_back(Record{kindString, kernel->name,
            //                   (unsigned long long) (kernel->start),
            //                   (unsigned long long) (kernel->end),
            //                   kernel->deviceId, kernel->streamId, kernel->correlationId});

            fprintf(log, "%s \"%s\" [ %llu - %llu ] device=%u context=%u stream=%u correlation=%u duration=%llu\n\n",
                kindString,
                kernel->name,
                (unsigned long long) (kernel->start),
                (unsigned long long) (kernel->end),
                kernel->deviceId, kernel->contextId, kernel->streamId, kernel->correlationId,
                (unsigned long long) (kernel->end - kernel->start));
            fprintf(log, "    grid=[%u,%u,%u], block=[%u,%u,%u], shared memory (static %u, dynamic %u)\n\n",
                kernel->gridX, kernel->gridY, kernel->gridZ,
                kernel->blockX, kernel->blockY, kernel->blockZ,
                kernel->staticSharedMemory, kernel->dynamicSharedMemory);
            break;
        }
        default:
            fprintf(log, "    Unknown Activity %d\n\n", record.kind);
    }
}

void Tracer::print_trace() {
    fprintf(stderr, "%lu records\n", records_.size());
    for (int i = 0; i < records_.size(); ++ i) {
        auto record = records_[i];
        fprintf(stderr, "%s %s [%llu - %llu]", record.kind, record.name, record.start, record.end);
    }
}

} // end of namespace cupti_tracer

