// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
/**
 * Stub library that can load other libraries for use in as DaCe programs
 **/

#ifdef _WIN32
    #include <windows.h>
    #define DACE_EXPORTED extern "C" __declspec(dllexport)
#else
    #include <dlfcn.h>
    #define DACE_EXPORTED extern "C"
#endif

// Workaround (see unload_library)
#include <omp.h>

// Loads a library and returns a handle to it, or NULL if there was an error
// NOTE: On Windows, path must be given as a Unicode string (UTF-16, or 
//       ctypes.c_wchar_p)
DACE_EXPORTED void *load_library(const char *filename) {
    if (!filename)
        return nullptr;

    void *hLibrary = nullptr;

#ifdef _WIN32
    hLibrary = (void *)LoadLibraryW((const wchar_t*)filename);
#else
    hLibrary = dlopen(filename, RTLD_LOCAL | RTLD_NOW);
#endif

    return hLibrary;
}

// Returns 1 if the library is already loaded, 0 if not, or -1 on error
DACE_EXPORTED int is_library_loaded(const char *filename) {
    if (!filename)
        return -1;

    void *hLibrary = nullptr;

#ifdef _WIN32
    hLibrary = (void *)GetModuleHandleW((const wchar_t*)filename);
#else
    hLibrary = dlopen(filename, RTLD_LOCAL | RTLD_NOW | RTLD_NOLOAD);
#endif

    if (hLibrary)
        return 1;
    return 0;
}

// Loads a library function and returns a pointer, or NULL if it was not found
DACE_EXPORTED void *get_symbol(void *hLibrary, const char *symbol) {
    if (!hLibrary || !symbol)
        return nullptr;

    void *sym = nullptr;

#ifdef _WIN32
    sym = GetProcAddress((HMODULE)hLibrary, symbol);
#else
    sym = dlsym(hLibrary, symbol);
#endif

    return sym;
}

// Loads a library and returns a handle to it, or NULL if there was an error
// NOTE: On Windows, path must be given as a Unicode string (UTF-16, or 
//       ctypes.c_wchar_p)
DACE_EXPORTED void unload_library(void *hLibrary) {
    if (!hLibrary)
        return;
    
    // Workaround so that OpenMP does not go ballistic when calling dlclose()
    omp_get_max_threads();

#ifdef _WIN32
    FreeLibrary((HMODULE)hLibrary);
#else
    dlclose(hLibrary);
#endif
}


