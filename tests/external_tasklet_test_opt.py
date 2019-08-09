diode.OpenPythonFile("external_tasklet.py")
diode.ChangeSDFGNodeProperties("s0_2", "language", "CPP")
diode.ChangeSDFGNodeProperties(
    "s0_2", "code", "b = a;\nstd::cout << \"I have been injected "
    "as raw C++ code! :-)\\n\";\n")
diode.ChangeSDFGNodeProperties("s0_2", "code_global", "#include <iostream>\n")
diode.Run(fail_on_nonzero=True)
