def program(backend, train_data, train_label,test_data,test_label, max_features,max_iter_qsvc,noise_model, user_messenger):
    from qiskit.utils import QuantumInstance
    from sklearn.decomposition import PCA
    from qiskit.circuit.library import PauliFeatureMap
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    import time

    result_dict = {}
    times = []
    scores = []
    iter_time = []
    qinst = QuantumInstance(backend, noise_model=noise_model)
    for n_features in range(2,max_features+1,2):
        iter_start = time.time()
        user_messenger.publish({"s" : "Iteration {} started, {} iterations will be executed in total".format(n_features,max_features)})
        pca = PCA(n_components=n_features)
        pca_start = time.time()
        train_data_pca = pca.fit_transform(train_data)
        pca_end = time.time()
        user_messenger.publish({"s":"PCA fitting of training data for iteration {} has been executed in {}s".format(n_features,pca_end-pca_start)})
        image_feature_map = PauliFeatureMap(feature_dimension=n_features)
        kernel = QuantumKernel(feature_map = image_feature_map, quantum_instance=qinst)
        qsvc = QSVC(quantum_kernel=kernel, max_iter = max_iter_qsvc)
        user_messenger.publish({"s" : "Fitting is starting for iteration {}".format(n_features)})
        start = time.time()
        qsvc.fit(train_data_pca,train_label)
        end = time.time()
        user_messenger.publish({"s": "Fitting has ended for iteration {} in {}s".format(n_features,end-start)})
        pca_test_start = time.time()
        test_data_pca = pca.fit_transform(test_data)
        pca_test_stop = time.time()
        user_messenger.publish({"s":"PCA fitting of testing data for iteration {} has been executed in {}s".format(n_features, pca_test_stop-pca_test_start)})
        start_score = time.time()
        s = qsvc.score(test_data_pca,test_label)
        end_score = time.time()
        user_messenger.publish({"s" : "Scoring for iteration {} ended in {}s".format(n_features,end_score-start_score)})
        times.append(end-start)
        scores.append(s)
        iter_stop = time.time()
        iter_time.append(iter_stop-iter_start)
        user_messenger.publish({"s":"Iteration {} ended in {}s".format(n_features,iter_stop-iter_start)})
    result_dict["times"] = times
    result_dict["scores"] = scores
    result_dict["iter_time"] = iter_time
    return(result_dict)

def main(backend,user_messenger,**kwargs):
    train_data = kwargs['train_data']
    train_label = kwargs['train_label']
    test_data = kwargs['test_data']
    test_label = kwargs['test_label']
    max_features = kwargs['max_features']
    max_iter_qsvc = kwargs['max_iter_qsvc']
    try:
        noise_model = kwargs['noise_model']
    except:
        noise_model = None
    result_dict = program(backend,train_data,train_label,test_data,test_label,max_features,max_iter_qsvc,noise_model,user_messenger)
    return(result_dict)