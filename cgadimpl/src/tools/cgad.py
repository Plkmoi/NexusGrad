import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import cgadimpl

print("cgadimpl loaded:", cgadimpl.hello())
