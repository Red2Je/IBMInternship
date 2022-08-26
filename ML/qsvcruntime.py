def program(backend, train_data, train_label, test_data, test_label, user_messenger):
    import numpy as np
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
    from qiskit_machine_learning.algorithms.classifiers import QSVC
    from qiskit.utils import QuantumInstance
    from qiskit_machine_learning.kernels import QuantumKernel

    feature_map_list = [TwoLocal(num_qubits = 5, reps = 4, rotation_blocks=['ry','rz'], entanglement_blocks='cz',parameter_prefix='in'),
                    RealAmplitudes(5, reps = 9,parameter_prefix='in'),
                    EfficientSU2(num_qubits = 5, reps = 4,parameter_prefix='in'),
                   ]
    result_dict = {}
    for feature_map in feature_map_list:
        kernel = QuantumKernel(feature_map=feature_map, quantum_instance=QuantumInstance(backend))
        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(train_data,train_label)
        train_score = qsvc.score(train_data,train_label)
        test_score = qsvc.score(test_data,test_label)
        result_dict[feature_map.name] = (train_score,test_score)
    
    return(result_dict)

def main(backend,user_messenger,**kwargs):
    train_data = kwargs['train_data']
    train_label = kwargs['train_label']
    test_data = kwargs['test_data']
    test_label = kwargs['test_label']
    result_dict = program(backend,train_data,train_label,test_data,test_label,user_messenger)
    return(result_dict)