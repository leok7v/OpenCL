#include "rt.h"
#include <assert.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>


#include "gpu.h"


// TODO: test 1..16 all types, test permutations of offset and shift, test limited max_items = 4, max_groups = 2, test huge, test performance

static void test(gpu_t* g) {
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
        fp32_t dot = gpu.dot_fp32(&m0, 0, 1, &m1, 0, 1, n);
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


#if 0
static void icd() { // Installable Client Drivers
    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_icd-opencl
    DISPLAY_DEVICE adapterDD = {0};
    adapterDD.cb = sizeof(DISPLAY_DEVICE);
    // DISPLAY_DEVICE associated with monitors.
    DISPLAY_DEVICE monitorDD = {0};
    monitorDD.cb = sizeof(DISPLAY_DEVICE);
    DWORD adapterIndex = 0;
    // Iterate over each adapter.
    while (EnumDisplayDevicesA(null, adapterIndex, &adapterDD, 0)) {
        if (adapterDD.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP) {
            traceln("%s", adapterDD.DeviceName);
            traceln("%s", adapterDD.DeviceKey + 18);
            // starts with
            // \Registry\Machine\System\CurrentControlSet\Control\Video\{3EA2EA3B-787B-11ED-9241-806E6F6E6963}\0000)
            // \Registry\Machine\
            // 012345678901234567
            /// OpenCLDriverName:
            // C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_9eaeaf7bfb6c744b\igdrcl64.dll
            HKEY key = null;
            DWORD valueType;
            char valueData[256];
            DWORD valueSize = sizeof(valueData);
            RegOpenKeyExA(HKEY_LOCAL_MACHINE, adapterDD.DeviceKey + 18, 0, KEY_READ, &key);
            RegQueryValueExA(key, "OpenCLDriverName", null, &valueType, (byte_t*)valueData, &valueSize);
            traceln("Value: %s\n", valueData);
            // C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_9eaeaf7bfb6c744b\igdrcl64.dll
            RegCloseKey(key);
            void* module = LoadLibraryA(valueData);

// 00031CE0 GTPin_Init
// 001C7C80 clEnqueueMarkerWithSyncObjectINTEL
// 001C7C90 clGetCLEventInfoINTEL
// 001C7F30 clGetCLObjectInfoINTEL
// 00027DE0 clGetExtensionFunctionAddress
// 0002B5F0 clGetPlatformInfo
// 001C7F40 clReleaseGlSharedEventINTEL
cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret)

            if (1) return;
            typedef int (*clGetPlatformIDs_t)(uint32_t num_entries,
                void* platforms, uint32_t* num_platforms);
            clGetPlatformIDs_t clGetPlatformIDs = (clGetPlatformIDs_t)GetProcAddress(module, "clIcdGetPlatformIDsKHR");
            #define CL_PLATFORM_ICD_SUFFIX_KHR 0x0920
            void* ids[16];
            uint32_t count = 0;
            int r = clGetPlatformIDs(16, ids, &count);
            typedef int (*clGetDeviceInfo_t)(void* device_id,
                        uint32_t  param_name,
                        size_t    param_value_size,
                        void *    param_value,
                        size_t *  param_value_size_ret);
clGetDeviceInfo_t clGetDeviceInfo = (clGetDeviceInfo_t)GetProcAddress(module, "clGetDeviceInfo");
            char dll_pathname[260];
            r = clGetDeviceInfo(ids[0], CL_PLATFORM_ICD_SUFFIX_KHR, countof(dll_pathname), dll_pathname, null);
            traceln("");
//           DWORD monitorIndex = 0;
//          // Iterate over each display connected to the
//          // current adapter.
//          while (EnumDisplayDevices(adapterDD.DeviceName, monitorIndex, &monitorDD, 0)) {
//              if (monitorDD.StateFlags & DISPLAY_DEVICE_ATTACHED) {
//                  // Create the display and store it in the internal list of displays.
//                  // Also, store a reference to the display in the display map based
//                  // on its type.
//                  Display* display = new Display(handle, getNumDisplays() + 1, adapterIndex, &adapterDD, monitorIndex, &monitorDD);
//                  displays.push_back(display);
//                  display_map[display->getType()].push_back(display);
//              }
//              monitorIndex++;
//          }
        }
        adapterIndex++;
    }
    HDEVINFO hDevInfo = null;
    SP_DEVINFO_DATA devInfoData = {sizeof(SP_DEVINFO_DATA)};
    DWORD devIndex = 0;
    // Create a device information set for display adapters
    hDevInfo = SetupDiGetClassDevs(&GUID_DEVCLASS_ADAPTER, NULL, NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (hDevInfo == INVALID_HANDLE_VALUE) {
        traceln("Failed to get device information set: %d\n", GetLastError());
        return;
    }
    // Enumerate through the devices in the set
    devInfoData.cbSize = sizeof(SP_DEVINFO_DATA);
    while (SetupDiEnumDeviceInfo(hDevInfo, devIndex, &devInfoData)) {
        TCHAR devInstanceId[1024];
        // Get the device instance ID
        if (!SetupDiGetDeviceInstanceId(hDevInfo, &devInfoData, devInstanceId, sizeof(devInstanceId), NULL)) {
            traceln("Failed to get device instance ID: %d\n", GetLastError());
            SetupDiDestroyDeviceInfoList(hDevInfo);
            return;
        }
        // Print the device instance ID
        traceln("Device Instance ID: %s", devInstanceId);
        // Increment the device index
        devIndex++;
    }
    if (GetLastError() != ERROR_NO_MORE_ITEMS) {
        traceln("Failed to enumerate devices: %d\n", GetLastError());
        SetupDiDestroyDeviceInfoList(hDevInfo);
        return;
    }
    SetupDiDestroyDeviceInfoList(hDevInfo);
}
#endif


int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
//  icd();
    bool profile = true;
    void* code = null;
    int64_t bytes = 0;
    int r = memmap_resource("gpu_cl", &code, &bytes);
    fatal_if(r != 0 || code == null || bytes == 0, "dot_cl is not in dot.rc");
    ocl.init();
    for (int cycles = 1; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            gpu_t gp = { 0 };
            ocl_context_t c = ocl.open(i, profile);
            traceln("%s\n", ocl.devices[i].name);
            gpu.init(&gp, &c, code, (int32_t)bytes);
            test(&gp);
//          traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            gpu.fini(&gp);
            ocl.close(&c);
        }
    }
    return 0;
}
