def program(backend, train_data, train_label,test_data,test_label,max_iter_qsvc, user_messenger):
    from qiskit.utils import QuantumInstance
    from sklearn.decomposition import PCA
    from qiskit.circuit.library import PauliFeatureMap
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    import time

    result_dict = {}
    n_features = 2
    user_messenger.publish({"s":"Started iterations for {} images".format(len(train_data))})
    qinst = QuantumInstance(backend)
    pca = PCA(n_components=n_features)
    train_data_pca = pca.fit_transform(train_data)
    image_feature_map = PauliFeatureMap(feature_dimension=n_features)
    kernel = QuantumKernel(feature_map = image_feature_map, quantum_instance=qinst)
    qsvc = QSVC(quantum_kernel=kernel, max_iter = max_iter_qsvc)
    start = time.time()
    qsvc.fit(train_data_pca,train_label)
    stop = time.time()
    test_data_pca = pca.fit_transform(test_data)
    s = qsvc.score(test_data_pca,test_label)
    user_messenger.publish({"s":"Ended iterations for {} images".format(len(train_data))})
    result_dict["time"] = stop-start
    result_dict["score"] = s
    return(result_dict)

def main(backend,user_messenger,**kwargs):
    train_data = kwargs['train_data']
    train_label = kwargs['train_label']
    test_data = kwargs['test_data']
    test_label = kwargs['test_label']
    max_iter_qsvc = kwargs['max_iter_qsvc']
    result_dict = program(backend,train_data,train_label,test_data,test_label,max_iter_qsvc,user_messenger)
    return(result_dict)