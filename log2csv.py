import csv

input_filename = "result3.log"
output_filename = f"{input_filename.split('.')[0]}.csv"
var_1_name = "lr"
var_2_name = "epochs"

with open(input_filename, "r") as file:
    data = file.read()

lines = data.strip().split("\n")
results = []
current_result = None

for line in lines:
    if line.startswith("RTraining"):
        if current_result:
            results.append(current_result)
        parts = line.split()
        var_1_value = parts[1].split(":")[1]
        var_2_value = parts[3].split(":")[1]
        current_result = {
            f"{var_1_name}": var_1_value,
            f"{var_2_name}": var_2_value,
            "Val": {"1-shot": None, "5-shot": None},
            "Test": {"1-shot": None, "5-shot": None},
        }
    elif line in ["Val", "Test"]:
        current_phase = line
    elif "shot1_acc" in line:
        acc = line.split()[1]
    elif "shot1_conf" in line:
        conf = line.split()[1]
        current_result[current_phase]["1-shot"] = f"{acc} ± {conf}"
    elif "shot5_acc" in line:
        acc = line.split()[1]
    elif "shot5_conf" in line:
        conf = line.split()[1]
        current_result[current_phase]["5-shot"] = f"{acc} ± {conf}"

if current_result:
    results.append(current_result)

with open(output_filename, "w", newline="") as csvfile:
    fieldnames = [
        "Phase",
        f"{var_1_name}",
        f"{var_2_name}",
        "1-shot (Accuracy ± Confidence)",
        "5-shot (Accuracy ± Confidence)",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        for phase in ["Val", "Test"]:
            writer.writerow(
                {
                    "Phase": phase,
                    f"{var_1_name}": result[var_1_name],
                    f"{var_2_name}": result[var_2_name],
                    "1-shot (Accuracy ± Confidence)": result[phase]["1-shot"],
                    "5-shot (Accuracy ± Confidence)": result[phase]["5-shot"],
                }
            )

print(f"Data has been written to {output_filename}")
