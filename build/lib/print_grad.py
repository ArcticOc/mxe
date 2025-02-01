def print_grad(tensor):
    tensor_flattened = tensor.flatten()

    tensor_list = tensor_flattened.tolist()

    output_string = ", ".join(map(str, tensor_list))

    with open("grads.csv", "a") as f:
        f.write(f"{output_string}\n")
