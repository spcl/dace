cloudsc_path='/users/msamuel/dwarf-p-cloudsc-dace'
dace_signature_file='full_cloudsc_logs/signature_dace_cloudscexp3.txt'

echo "Using signature file at $dace_signature_file"

parameters=$(
    cat $dace_signature_file |
    tr ',' '\n' |
    rev | cut -d ' ' -f1 | rev | 
    tr "\n" "," )
        # | sed --expression="s/%/\&\n,\&/g")
# remove last comma
parameters=${parameters::-1}
# echo $parameters
# put parameter names inside the fortran driver file
sed "s/<signature>/$parameters\&/g" "$cloudsc_path/src/cloudsc_fortran/cloudsc_driver_mod.F90.template" \
    > "$cloudsc_path/src/cloudsc_fortran/cloudsc_driver_mod.F90"

# create parameter list where all parameters are points
# cat $dace_signature_file
# echo ""
parameters_scalar=$(cat $dace_signature_file | grep -oE "(int|double) [a-Z0-9_]+(, )?")
parameters_array=$(cat $dace_signature_file | grep -oE "(int|double) \* __restrict__ [a-Z0-9_]+(, )?")
# echo "$parameters_scalar"
# echo "Parametrers array"
# echo "$parameters_array"
parameters_pointer=$(echo $parameters_scalar |
    sed --expression='s/double/double \*/g' |
    sed --expression='s/int/int \*/g')
parameters_pointer=$(echo "$parameters_array $parameters_pointer")
# echo "$parameters_pointer"
parameters_dereference=$(echo $parameters_scalar |
    sed --expression='s/int /\*/g' |
    sed --expression='s/double /\*/g')
# echo "$parameters_dereference"
parameters_array_names=$(echo "$parameters_array" | cut -d ' ' -f 4)
# echo "Paramters array names"
# echo "$parameters_array_names"
parameters_dereference=$(echo "$parameters_array_names $parameters_dereference")
# echo "$parameters_dereference"


adapter_file="$cloudsc_path/src/cloudsc_fortran/working.cc"
echo "#include <cstdlib>" > $adapter_file 
echo "#include <dace/dace.h>" >> $adapter_file
echo "#include <chrono>" >> $adapter_file
echo "typedef void * CLOUDSCOUTERHandle_t;" >> $adapter_file
echo "extern \"C\" CLOUDSCOUTERHandle_t __dace_init_CLOUDSCOUTER();" >> $adapter_file
echo "extern \"C\" void __dace_exit_CLOUDSCOUTER(CLOUDSCOUTERHandle_t handle);" >> $adapter_file
echo "extern \"C\" void __program_CLOUDSCOUTER(CLOUDSCOUTERHandle_t handle, $(cat $dace_signature_file));" >> $adapter_file
echo "" >> $adapter_file
echo "void cloudsc2_internal($(cat $dace_signature_file))" >> $adapter_file
echo "{" >> $adapter_file
echo "    CLOUDSCOUTERHandle_t handle;" >> $adapter_file
echo "    handle = __dace_init_CLOUDSCOUTER();" >> $adapter_file
echo "    auto start = std::chrono::high_resolution_clock::now();" >> $adapter_file
echo "    __program_CLOUDSCOUTER(handle, $parameters);" >> $adapter_file
echo "    auto end = std::chrono::high_resolution_clock::now();" >> $adapter_file
echo "    printf(\"Time %d [ms]\n\", std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());" >> $adapter_file
echo "    __dace_exit_CLOUDSCOUTER(handle);" >>  $adapter_file
echo "}" >> $adapter_file
echo "DACE_EXPORTED void cloudsc2_($parameters_pointer)" >> $adapter_file
echo "{" >> $adapter_file
echo "    cloudsc2_internal($parameters_dereference);" >> $adapter_file
echo "}" >> $adapter_file
