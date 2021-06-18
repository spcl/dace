#small utility to run hw_emu tests without killing xrt

#vadd
touch tmp_hw_emu_run_helper.py
echo 'from hbm_vadd_fpga import exec_test \nexec_test(1, 50, 2, "vadd_2b1d")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_vadd_fpga import exec_test \nexec_test(2, 50, 2, "vadd_2b2d")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_vadd_fpga import exec_test \nexec_test(3, 10, 2, "vadd_2b3d")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_vadd_fpga import exec_test \nexec_test(1, 50, 8, "vadd_8b1d", True)' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

#reduce

echo 'from hbm_reduce_fpga import exec_test \nexec_test(2, 3, 2, "red_2x3_2b")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_reduce_fpga import exec_test \nexec_test(10, 50, 4, "red_10x50_4b")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_reduce_fpga import exec_test \nexec_test(1, 50, 1, "red_1x50_1b")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_reduce_fpga import exec_test \nexec_test(1, 40, 8, "red_1x40_8b")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'from hbm_reduce_fpga import exec_test \nexec_test(2, 40, 6, "red_2x40_6b")' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

python3 hbm_dynamic_memlets.py

python3 hbm_deeply_nested_fpga.py

#copy test

echo 'import hbm_copy_fpga \nhbm_copy_fpga.check_host2copy1()' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'import hbm_copy_fpga \nhbm_copy_fpga.check_dev2host1()' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'import hbm_copy_fpga \nhbm_copy_fpga.check_dev2dev1()' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'import hbm_copy_fpga \nhbm_copy_fpga.check_hbm2hbm1()' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py

echo 'import hbm_copy_fpga \nhbm_copy_fpga.check_hbm2ddr1()' > tmp_hw_emu_run_helper.py
python3 tmp_hw_emu_run_helper.py