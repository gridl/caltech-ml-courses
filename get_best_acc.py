import os
import json


results_list = []
main_dir = 'logs'
models = [model for model in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, model))]
for model in models:
    model_dir = os.path.join(main_dir, model)
    model_runs = [run for run in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, run))]
    for run in model_runs:
        run_folder = os.path.join(model_dir, run)
        try:
            with open(os.path.join(run_folder, 'logs.json'), 'r') as f:
                logs = json.load(f)
        except FileNotFoundError:
            continue
        accuracies = [acc[1] for acc in logs['val']['accuracy']]
        max_acc = max(accuracies)
        results_list.append((max_acc, '%s_run_%s' % (model, run)))

results_list.sort(reverse=True)
for res in results_list:
    print("Acc: %f, model: %s" % (res[0], res[1]))
