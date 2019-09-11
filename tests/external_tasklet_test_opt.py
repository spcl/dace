diode.OpenPythonFile("external_tasklet.py")
diode.ChangeSDFGNodeProperties(
    "s0_2",
    "code",
    """
    {
        "language": "CPP",
        "string_data": "b = a; std::cout << \\"I have been injected as raw C++ code! :-)\\\\n\\";"
    }
    """,
    json=True)
diode.ChangeSDFGNodeProperties("s0_2", "code_global", "#include <iostream>\n")
diode.Run(fail_on_nonzero=True)
