import dace


repldict = {"_for_it_88": "i"}
sdfg = dace.SDFG.from_file("t_before.sdfg")
sdfg.replace_dict(repldict)
sdfg.save("t_after.sdfg")