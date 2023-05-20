//#include <math.h>
//#include <malloc.h>
#include "rt.h"
#include "gpu.h"
#include "blast.h"


// TODO: test 1..16 all types, test permutations of offset and shift, test limited max_items = 4, max_groups = 2, test huge, test performance

static void test(blast_t* b) {
    gpu_t* g = b->gpu;
    double err = 0;
    for (int n = 2; n < 17; n++) {
        const int64_t bytes = n * sizeof(fp32_t);
        gpu_memory_t m0 = gpu.allocate(g, gpu_allocate_write, bytes);
        gpu_memory_t m1 = gpu.allocate(g, gpu_allocate_write, bytes);
        fp32_t* x = (fp32_t*)gpu.map(&m0, gpu_map_write, 0, bytes);
        fp32_t* y = (fp32_t*)gpu.map(&m1, gpu_map_write, 0, bytes);
        fp32_t sum = 0;
        for (int32_t i = 0; i < n; i++) {
//          x[i] = 1.0f + (i % 2 == 0 ? -1.0f : +1.f) / (i + 1);
//          y[i] = 1.0f - (i % 2 == 0 ? +1.0f : -1.f) / (i + 1);
x[i] = (fp32_t)(i + 1);
y[i] = (fp32_t)(i + 1);
//          traceln("%f %f %f", x[i], y[i], x[i] * y[i]);
            sum += x[i] * y[i];
        }
        gpu.unmap(&m1);
        gpu.unmap(&m0);
        fp64_t dot = b->dot_fp32(b, &m0, 0, 1, &m1, 0, 1, n);
        gpu.deallocate(&m0);
        gpu.deallocate(&m1);
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

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
//  icd();
    bool profile = true;
    ocl.init();
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            ocl_context_t c = ocl.open(i, profile);
            traceln("%s\n", ocl.devices[i].name);
            gpu_t g = { 0 };
            gpu.init(&g, &c);
            blast_t b = { 0 };
            blast.init(&b, &g);
            test(&b);
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            blast.fini(&b);
            gpu.fini(&g);
            ocl.close(&c);
        }
    }
    return 0;
}
