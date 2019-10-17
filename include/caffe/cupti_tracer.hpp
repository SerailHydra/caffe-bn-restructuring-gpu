#include <cstdio>
#include <vector>
#include <stdlib.h>
#include <memory>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <cupti.h>

namespace cupti_tracer {

class CUptiManager;

class Tracer {
    friend class CUptiManager;
    public:
        struct Record {
            const char *kind;
            const char *name;
            uint64_t start, end; // start & end time of the record in ns
            uint32_t processId, threadId; // deviceID & streamID for a kernel record
            uint32_t correlationId;
        };

        Tracer();
        void start();
        void stop();
        void print_trace();
        std::vector<Record> get_records() {
            return records_;
        }

    private:
        static void CUPTIAPI APICallback(void *userdata, CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid, const void *cbdata);

        void ActivityCallback(const CUpti_Activity &record);

    private:
        CUpti_SubscriberHandle subscriber_;
        std::vector<Record> records_;
        CUptiManager *manager_;
}; // end of class Tracer

class CUptiManager {
    public:
        static CUptiManager *Get();
        void Initialize(Tracer *tracer);

    private:
        static void BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
            Get()->InternalBufferRequested(buffer, size, maxNumRecords);
        }

        static void BufferCompleted(CUcontext ctx, uint32_t streamId,
                                    uint8_t *buffer, size_t size, size_t validSize) {
            Get()->InternalBufferCompleted(ctx, streamId, buffer, size, validSize);
        }

        void InternalBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
        void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                     uint8_t *buffer, size_t size, size_t validSize);

        Tracer *tracer_;
}; // end of class CUptiManager

std::shared_ptr<Tracer> _GetSharedRef() {
    static std::shared_ptr<Tracer> inst(new Tracer());
    return inst;
}

Tracer* Get() {
    static Tracer *ptr = _GetSharedRef().get();
    return ptr;
}

} // end of namespace cupti_tracer

void startCUptiTracing() {
    static auto tracer = cupti_tracer::Get();
    tracer->start();
}

void endCUptiTracing() {
    static auto tracer = cupti_tracer::Get();
    tracer->stop();
}

