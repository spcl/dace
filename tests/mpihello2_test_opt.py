# Embarassingly parallel MPI code, uses Immaterial storage
diode.OpenPythonFile("mpihello2.py")
result = diode.Run()
diode.ExpandNode("MPITransformMap")
diode.Run()
