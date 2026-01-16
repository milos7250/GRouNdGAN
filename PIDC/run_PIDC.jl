using NetworkInference
dataset_name = string(ARGS[1])
out_file_path = string(ARGS[2])

@time network = infer_network(dataset_name, PIDCNetworkInference(), out_file_path=out_file_path);