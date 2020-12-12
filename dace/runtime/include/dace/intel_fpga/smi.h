#pragma once

/**
    Includes and utility functions for SMI backend
*/


#include <mpi.h>

void checkMpi(int code, const char* location, int line)
{
    if (code != MPI_SUCCESS)
    {
        char error[256];
        int length;
        MPI_Error_string(code, error, &length);
        std::cerr << "MPI error at " << location << ":" << line << ": " << error << std::endl;
    }
}

#define CHECK_MPI(err) checkMpi((err), __FILE__, __LINE__);
