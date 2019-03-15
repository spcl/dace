# Embarassingly parallel MPI code, uses Immaterial storage
diode.OpenPythonFile("mpihello.py")
result = diode.Run()
diode.ExpandNode("MPITransformMap")
diode.Run()
