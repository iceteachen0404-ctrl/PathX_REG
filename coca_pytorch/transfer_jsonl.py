import json

input_file = "./inference_outputs_final/checkpoints0819_multibatch_train_seed60.jsonl"
output_file = "./inference_outputs_final/checkpoints0819_multibatch_train_seed60.json"

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)

            # 在第一个分号 ';' 后加换行符
            report = obj["generated"]
            if ";" in report:
                report = report.replace(";", ";\n", 1)

            data.append({
                "id": obj["filename"].replace(".h5", ".tiff"),
                "report": report
            })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"转换完成！输出：{output_file}")
