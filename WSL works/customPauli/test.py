from custom import CustomPauliFeatureMap
from qiskit.circuit import Parameter, ParameterVector
print("Float test")
qc = CustomPauliFeatureMap(2)
print("Float test passed")
print("Parameter test")
qc = CustomPauliFeatureMap(2,alpha = Parameter("t"))
# print(qc.decompose().draw())
print("Parameter test passed")
print("ParameterVector test")
pv = ParameterVector("b",6)
qc = CustomPauliFeatureMap(2, alpha = pv)
print("ParameterVector test passed")
print("Final : ")
print(qc.decompose().draw())
value_list = range(1,7)
dict_bind = {pv[i] : value_list[i] for i in range(2,6)}
binded = qc.bind_parameters(dict_bind)
# print(binded.decompose().draw())