#include "rt.h"
#include "blast.h"
#include "blast.h"
#include <CL/opencl.h>

// TODO: test 1..16 all types, test permutations of offset and shift, test limited max_items = 4, max_groups = 2, test huge, test performance

static uint32_t seed;

static size_t sizes[] = { sizeof(fp16_t), sizeof(fp32_t), sizeof(fp64_t) };

typedef struct test_dot_s {
    int64_t bytes0;
    int64_t bytes1;
    blast_memory_t v0;
    blast_memory_t v1;
    void* a0;
    void* a1;
    double expected;
    double dot;
    double rse; // root square error
} test_dot_t;

static void test_dot_alloc(blast_t* b, test_dot_t* td, int fpp,
        int64_t n0, int64_t n1) {
    td->bytes0 = n0 * sizes[fpp];
    td->bytes1 = n1 * sizes[fpp];
    td->v0 = blast.allocate(b, blast_access_write, td->bytes0);
    td->v1 = blast.allocate(b, blast_access_write, td->bytes1);
}

static void test_dot_map(test_dot_t* td) {
    td->a0 = blast.map(&td->v0, blast_access_write, 0, td->bytes0);
    td->a1 = blast.map(&td->v1, blast_access_write, 0, td->bytes1);
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
        int64_t o0, int64_t s0, int64_t o1, int64_t s1, bool verbose) {
    assert(1 <= n && n <= 16);
    assert(o0 >= 0 && s0 >= 1 && o1 >= 0 && s1 >= 1);
    #pragma push_macro("at0")
    #pragma push_macro("at1")
    #define at0(type, i) ((type*)td.a0 + o0 + i * s0)
    #define at1(type, i) ((type*)td.a1 + o1 + i * s1)
    test_dot_t td = {0};
    test_dot_alloc(b, &td, fpp, o0 + n * s0, o1 + n * s1);
    test_dot_map(&td);
    // init memory by garbage
    for (int i = 0; i < td.bytes0; i++) {
        *((byte_t*)td.a0 + i) = (byte_t)random32(&seed);
    }
    for (int i = 0; i < td.bytes1; i++) { // init memory by garbage
        *((byte_t*)td.a1 + i) = (byte_t)random32(&seed);
    }
    td.expected = 0;
    for (int i = 0; i < n; i++) {
        if (fpp == blast_fpp16) {
            *at0(fp16_t, i) = fp32to16((fp32_t)(i + 1));
            *at1(fp16_t, i) = fp32to16((fp32_t)(n - i));
        } else if (fpp == blast_fpp32) {
            *at0(fp32_t, i) = (fp32_t)(i + 1);
            *at1(fp32_t, i) = (fp32_t)(n - i);
        } else if (fpp == blast_fpp64) {
            *at0(fp64_t, i) = (fp64_t)(i + 1);
            *at1(fp64_t, i) = (fp64_t)(n - i);
        } else {
            fatal_if("fpp", "%d", fpp);
        }
        td.expected += (fp64_t)(i + 1) * (fp64_t)(n - i);
    }
    #pragma pop_macro("at1")
    #pragma pop_macro("at0")
    test_dot_unmap(&td);
    td.dot = 0;
    td.dot = b->dot[fpp](&td.v0, o0, s0, &td.v1, o1, s1, n);
    test_dot_free(&td);
    td.rse = td.expected - td.dot;
    td.rse = sqrt(td.rse * td.rse);
    if (verbose || td.rse > CL_DBL_EPSILON) {
        traceln("%s[%2d] [o:%2d s:%2d] [o:%2d s:%2d] "
                "%25.17f expected: %25.17f rse: %.17f",
                blast_fpp_names[fpp], n, o0, s0, o1, s1,
                td.dot, td.expected, td.rse);
    }
    fatal_if(td.rse > CL_DBL_EPSILON);
}

static void test_permutations() {
    ocl_override_t ov0 = {
        .max_groups = 2,
        .max_items = 4
    };
    for (int i = 0; i < ocl.count; i++) {
        ocl_context_t c = ocl.open(i, &ov0); // test on small groups/items
        traceln("%s\n", ocl.devices[i].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        for (int n = 1; n < 7; n++) {
            for (int fpp = blast_fpp16; fpp <= blast_fpp64; fpp++) {
                if (b.dot[fpp] != null) {
                    for (int o0 = 0; o0 < 3; o0++) {
                        for (int o1 = 0; o1 < 3; o1++) {
                            for (int s0 = 1; s0 < 3; s0++) {
                                for (int s1 = 1; s1 < 3; s1++) {
                                    test_first_n(&b, n, fpp, o0, s0, o1, s1, false);
                                }
                            }
                        }
                    }
                }
            }
        }
        blast.fini(&b);
        ocl.close(&c);
    }
    ocl_override_t ov1 = {
        .max_groups = 2,
        .max_items = 3
    };
    for (int i = 0; i < ocl.count; i++) {
        ocl_context_t c = ocl.open(i, &ov1); // test on small groups/items
        traceln("%s\n", ocl.devices[i].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        for (int n = 1; n < 11; n++) {
            for (int fpp = blast_fpp16; fpp <= blast_fpp64; fpp++) {
//              traceln("%s n: %d", blast_fpp_names[fpp], n);
                if (b.dot[fpp] != null) {
                    for (int o0 = 0; o0 < 4; o0++) {
                        for (int o1 = 0; o1 < 4; o1++) {
                            for (int s0 = 1; s0 < 3; s0++) {
                                for (int s1 = 1; s1 < 3; s1++) {
                                    test_first_n(&b, n, fpp, o0, s0, o1, s1, false);
                                }
                            }
                        }
                    }
                }
            }
        }
        blast.fini(&b);
        ocl.close(&c);
    }
}

static void test_performance() {
//  enum { N = 16 * 1024 * 1024 + };  // TODO: fails at N = 16 * 1024 * 1024 + 1
    enum { N = 16 * 1024 * 1024 };
    static ocl_profiling_t p[16 * 1024]; // max 4K kernel invocations measurement
    ocl_override_t ov = {
        .profiling = p,
        .max_profiling_count = countof(p),
        .profiling_count = 0
    };
    // profiling measurement:
    for (int d = 0; d < ocl.count; d++) {
        ocl_context_t c = ocl.open(d, &ov);
        traceln("%s\n", ocl.devices[d].name);
        blast_t b = { 0 };
        blast.init(&b, &c);
        double err = 0;
        const int64_t bytes = N * sizeof(fp32_t);
        blast_memory_t m0 = blast.allocate(&b, blast_access_write, bytes);
        blast_memory_t m1 = blast.allocate(&b, blast_access_write, bytes);
        fp32_t* x = (fp32_t*)blast.map(&m0, blast_access_write, 0, bytes);
        fp32_t* y = (fp32_t*)blast.map(&m1, blast_access_write, 0, bytes);
        fp32_t delta = 1.0f / (1 << 20);
        delta *= delta;
        fp32_t sum = 0;
        for (int64_t i = 0; i < N; i++) {
            fp32_t sign = (i % 2 == 0 ? -1.0f : +1.f);
            x[i] = 1.0f + sign * (i * delta);
            y[i] = 1.0f - sign * (i * delta);
//          traceln("%f %f %f", x[i], y[i], x[i] * y[i]);
            sum += x[i] * y[i];
        }
//      traceln("sum: %15.7e delta: %15.7e", sum, delta);
        blast.unmap(&m1);
        blast.unmap(&m0);
        double host = seconds();
        fp64_t dot = b.dot[blast_fpp32](&m0, 0, 1, &m1, 0, 1, N);
        host = seconds() - host;
        blast.deallocate(&m0);
        blast.deallocate(&m1);
        double rse = sqrt(pow(dot - sum, 2)) / sum;
//      traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", N, dot, sum, rse);
        if (rse > err) {
            if (rse > CL_DBL_EPSILON) {
                traceln("n: %d dot: %.7f  sum: %.7F rse: %.7f\n", N, dot, sum, rse);
                traceln("n: %d dot: %.7e  sum: %.7e rse: %.7e\n", N, dot, sum, rse);
            }
            err = rse;
        }
        assert(fabs(dot - sum) <= CL_DBL_EPSILON, "dot_product(): %.7e != %.7e\n", dot, sum);
        traceln("dot_fp32 kernel x %lld: %.3f user: %.3f host: %.3f ms Gflops: %.6f",
            N, p->time * MSEC_IN_SEC, p->user * MSEC_IN_SEC, host * MSEC_IN_SEC, p->gflops);
//      traceln("max rse: %.7e %.17f\n", err, err);
        blast.fini(&b);
        // see: add.c for averaging
        ocl.close(&c);
    }
}

void dot_test();

static void dot_tests() {
//  dot_test();
//  test_permutations();
    test_performance();
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    ocl.init();
    dot_tests();
    return 0;
}
