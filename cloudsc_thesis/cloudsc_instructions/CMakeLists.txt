# (C) Copyright 1988- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Define this dwarf variant as an ECBuild feature
ecbuild_add_option( FEATURE CLOUDSC_FORTRAN
    DESCRIPTION "Build the new Fortran version CLOUDSC using Serialbox" DEFAULT ON
    CONDITION Serialbox_FOUND OR HDF5_FOUND
)

if( HAVE_CLOUDSC_FORTRAN )

    # Define the binary build target for this variant
    ecbuild_add_executable( TARGET dwarf-cloudsc-fortran
        SOURCES
            dwarf_cloudsc.F90
            cloudsc_driver_mod.F90
            cloudsc.F90
	        working.cc
        DEFINITIONS ${CLOUDSC_DEFINITIONS}
    )
    target_link_libraries( dwarf-cloudsc-fortran PRIVATE cloudsc-common-lib CLOUDSCOUTER4)
    target_link_directories(dwarf-cloudsc-fortran PRIVATE
        <PathToYourRepository>/cloudsc_thesis/.dacecache/CLOUDSCOUTER4/build)

    target_include_directories(dwarf-cloudsc-fortran PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(dwarf-cloudsc-fortran PRIVATE <PathToYourRepository>/dace/runtime/include ${CMAKE_CURRENT_BINARY_DIR}/../common)

    if( HAVE_OMP AND TARGET OpenMP::OpenMP_Fortran )
        target_link_libraries( dwarf-cloudsc-fortran PRIVATE OpenMP::OpenMP_Fortran )
    endif()

    if( HAVE_MPI AND TARGET MPI::MPI_Fortran )
        target_link_libraries( dwarf-cloudsc-fortran PRIVATE MPI::MPI_Fortran )
    endif()

    # Create symlink for the input data
    if( HAVE_SERIALBOX )
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/../../data ${CMAKE_CURRENT_BINARY_DIR}/../../../data )
    endif()

    if( HAVE_HDF5 )
      target_include_directories( dwarf-cloudsc-fortran PRIVATE ${HDF5_Fortran_INCLUDE_DIRS} )
      target_link_libraries( dwarf-cloudsc-fortran PRIVATE ${HDF5_LIBRARIES} )
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/../../config-files/input.h5 ${CMAKE_CURRENT_BINARY_DIR}/../../../input.h5 )
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_SOURCE_DIR}/../../config-files/reference.h5 ${CMAKE_CURRENT_BINARY_DIR}/../../../reference.h5 )
    endif()
endif()
