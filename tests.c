#include "rt.h"
#include "blast.h"
#include "blast.h"

// TODO: test 1..16 all types, test permutations of offset and shift, test limited max_items = 4, max_groups = 2, test huge, test performance

static uint32_t seed;

static size_t sizes[] = { sizeof(fp16_t), sizeof(fp32_t), sizeof(fp64_t) };

typedef struct test_dot_s {
    int64_t bytes;
    blast_memory_t v0;
    blast_memory_t v1;
    void* a0;
    void* a1;
    double expected;
    double dot;
    double rse; // root square error
} test_dot_t;

static void test_dot_alloc(blast_t* b, test_dot_t* td, int fpp, int64_t n) {
    td->bytes = n * sizes[fpp];
    td->v0 = blast.allocate(b, blast_access_write, td->bytes);
    td->v1 = blast.allocate(b, blast_access_write, td->bytes);
}

static void test_dot_map(test_dot_t* td) {
    td->a0 = blast.map(&td->v0, blast_access_write, 0, td->bytes);
    td->a1 = blast.map(&td->v1, blast_access_write, 0, td->bytes);
}

static void test_dot_unmap(test_dot_t* td) {
    blast.unmap(&td->v0);
    blast.unmap(&td->v1);
}

static void test_dot_free(test_dot_t* td) {
    blast.deallocate(&td->v0);
    blast.deallocate(&td->v1);
}

static void test_first_n(blast_t* b, int64_t n, int fpp,
        int64_t offset, int64_t stride, bool verbose) {
    assert(1 <= n && n <= 16);
    #pragma push_macro("at")
    #define at(type, f, i) ((type*)td.f + offset + i * stride)
    assert(offset >= 0 && stride >= 1);
    test_dot_t td = {0};
    test_dot_alloc(b, &td, fpp, offset + n * stride);
    test_dot_map(&td);
    for (int i = 0; i < td.bytes; i++) {
        *((byte_t*)td.a0 + i) = (byte_t)random32(&seed);
        *((byte_t*)td.a1 + i) = (byte_t)random32(&seed);
    }
    td.expected = 0;
    for (int i = 0; i < n; i++) {
        if (fpp == blast_fpp16) {
            *at(fp16_t, a0, i) = fp32to16((fp32_t)(i + 1));
            *at(fp16_t, a1, i) = fp32to16((fp32_t)(n - i));
        } else if (fpp == blast_fpp32) {
            *at(fp32_t, a0, i) = (fp32_t)(i + 1);
            *at(fp32_t, a1, i) = (fp32_t)(n - i);
        } else if (fpp == blast_fpp64) {
            *at(fp64_t, a0, i) = (fp64_t)(i + 1);
            *at(fp64_t, a1, i) = (fp64_t)(n - i);
        } else {
            fatal_if("fpp", "%d", fpp);
        }
        td.expected += (fp64_t)(i + 1) * (fp64_t)(n - i);
    }
    #pragma pop_macro("at")
    test_dot_unmap(&td);
    td.dot = 0;
    td.dot = b->dot[fpp](&td.v0, offset, stride, &td.v1, offset, stride, n);
    test_dot_free(&td);
    td.rse = td.expected - td.dot;
    td.rse = sqrt(td.rse * td.rse);
    if (verbose) {
        traceln("%s[%2d] %25.17f %25.17f rse: %.17f", blast_fpp_names[fpp], n,
            td.dot, td.expected, td.rse);
    }
}

static void test(blast_t* b) {
    blast_t* g = b;
    double err = 0;
    for (int n = 2; n < 17; n++) {
        const int64_t bytes = n * sizeof(fp32_t);
        blast_memory_t m0 = blast.allocate(g, blast_access_write, bytes);
        blast_memory_t m1 = blast.allocate(g, blast_access_write, bytes);
        fp32_t* x = (fp32_t*)blast.map(&m0, blast_access_write, 0, bytes);
        fp32_t* y = (fp32_t*)blast.map(&m1, blast_access_write, 0, bytes);
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) {
//          x[i] = 1.0f + (i % 2 == 0 ? -1.0f : +1.f) / (i + 1);
//          y[i] = 1.0f - (i % 2 == 0 ? +1.0f : -1.f) / (i + 1);
x[i] = (fp32_t)(i + 1);
y[i] = (fp32_t)(i + 1);
//          traceln("%f %f %f", x[i], y[i], x[i] * y[i]);
            sum += x[i] * y[i];
        }
        blast.unmap(&m1);
        blast.unmap(&m0);
        fp64_t dot = b->dot[blast_fpp32](&m0, 0, 1, &m1, 0, 1, n);
        blast.deallocate(&m0);
        blast.deallocate(&m1);
        double rse = sqrt(pow(dot - sum, 2));
        traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", n, dot, sum, rse);
        if (rse > err) {
//          traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", n, dot, sum, rse);
//          traceln("n: %d dot: %.7e  sum: %.7e rse: %.7e\n", n, dot, sum, rse);
            err = rse;
        }
//      assertion(fabs(dot - sum) <= CL_DBL_EPSILON, "dot_product():%.7e != %.7e\n", dot, sum);
    }
    traceln("max rse: %.7e %.17f\n", err, err);
}

static void test1() {
    ocl_override_t ov = {
        .max_groups = 2,
        .max_items = 4
    };
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            ocl_context_t c = ocl.open(i, &ov); // test on small groups/items
            traceln("%s\n", ocl.devices[i].name);
            blast_t b = { 0 };
            blast.init(&b, &c);
            for (int n = 1; n < 16; n++) {
                for (int fpp = blast_fpp16; fpp <= blast_fpp64; fpp++) {
                    if (b.dot[fpp] != null) {
                        test_first_n(&b, n, fpp, 0, 1, true);
                    }
                }
            }
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            blast.fini(&b);
            ocl.close(&c);
        }
    }
}

static void test2() {
    static ocl_profiling_t p[4096]; // max 4K kernel invocations measurement
    ocl_override_t ov = {
        .max_groups = 0,
        .max_items = 0,
        .profiling = p,
        .max_profiling_count = countof(p),
        .profiling_count = 0
    };
    // profiling measurement:
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            ocl_context_t c = ocl.open(i, &ov);
            traceln("%s\n", ocl.devices[i].name);
            blast_t b = { 0 };
            blast.init(&b, &c);
            for (int n = 1; n < 16; n++) {
                for (int fpp = blast_fpp16; fpp <= blast_fpp64; fpp++) {
                    if (b.dot[fpp] != null) {
                        // TODO: need different test  1.0 +/- very small delta e.g. DBL_EPSILON, FLT_EPSILON, FP16_EPSILON * i
                        test_first_n(&b, n, fpp, 0, 1, false);
                    }
                }
            }
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            blast.fini(&b);
            ocl.close(&c);
        }
    }
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    test1();
    test2();
    return 0;
}
