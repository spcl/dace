from dace.config import Config

Config.set("compiler", "cpu", "libs", value="papi", autosave=True)
