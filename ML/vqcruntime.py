def program(backend,train_data,train_label,test_data,test_label, user_messenger):
    import numpy as np
    cost_optimized = []
    def callback_parameter(weights,cost):
        cost_optimized.append(cost)
        user_messenger.publish({"cost" : cost_optimized})


    from qiskit.circuit.library import EfficientSU2,TwoLocal, ZZFeatureMap
    from qiskit.algorithms.optimizers import SPSA
    from qiskit.utils import QuantumInstance 
    from qiskit_machine_learning.algorithms.classifiers import VQC
    feature_map = EfficientSU2(num_qubits=2,reps = 4,parameter_prefix='in')
    ansatz = TwoLocal(num_qubits=feature_map.num_qubits,rotation_blocks=['ry','rz'],entanglement_blocks='cz')
    optimizer = SPSA(maxiter=100)
    vqc = VQC(feature_map=feature_map,
                ansatz=ansatz,
                loss='cross_entropy',
                optimizer=optimizer,
                quantum_instance=QuantumInstance(backend=backend),
                callback=callback_parameter)
    vqc.fit(train_data,train_label)
    train_score = vqc.score(train_data,train_label)
    test_score = vqc.score(test_data,test_label)
    return(train_score,test_score)


def main(backend,user_messenger,**kwargs):
    train_data = kwargs['train_data']
    train_label = kwargs['train_label']
    test_data = kwargs['test_data']
    test_label = kwargs['test_label']

    train_score,test_score = program(backend,train_data,train_label,test_data,test_label,user_messenger)
    return {'train_score':train_score,'test_score':test_score}
    
