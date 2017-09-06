"""Server to display logs for various models"""
import os
import json

from flask import Flask, render_template
import click


app = Flask(__name__)
graph_decl_f_name = 'graph_str.txt'
description_f_name = 'description.txt'
train_params_f_name = 'train_params.json'
data_params_f_name = 'data_params.json'
logs_f_name = 'logs.json'


class ModelParserException(Exception):
    pass


def parse_model(model_path):
    """
    Args:
        model_path(str): path to the model folder
    """
    model_name = os.path.basename(model_path)
    model_data = {}
    graph_decl_path = os.path.join(model_path, graph_decl_f_name)
    if not os.path.exists(graph_decl_path):
        raise ModelParserException(f"No such model: {model_path}")
    with open(graph_decl_path, 'r') as f:
        model_data['graph_decl'] = f.read()
    runs = [f for f in os.listdir(model_path) if
            os.path.isdir(os.path.join(model_path, f))]
    runs_data = []
    for run_no in runs:
        run_path = os.path.join(model_path, run_no)
        with open(os.path.join(run_path, description_f_name), 'r') as f:
            description = f.read()
        with open(os.path.join(run_path, train_params_f_name), 'r') as f:
            train_params = json.load(f)
        try:
            with open(os.path.join(run_path, data_params_f_name), 'r') as f:
                data_params = json.load(f)
        except Exception as e:
            data_params = {}
        with open(os.path.join(run_path, logs_f_name), 'r') as f:
            logs = json.load(f)
        run_dict = {
            'run_no': run_no,
            'description': description,
            'train_params': train_params,
            'data_params': data_params,
            'logs': logs
        }
        runs_data.append(run_dict)
    model_data["runs_data"] = runs_data
    return model_name, model_data


def parse_logs(logs_dir):
    models_dirs = [os.path.join(logs_dir, p) for p in os.listdir(logs_dir)
                   if os.path.isdir(os.path.join(logs_dir, p))]
    models_info = {}
    for model_path in models_dirs:
        try:
            model_name, model_info = parse_model(model_path)
        except Exception as e:
            raise e
            continue
        models_info[model_name] = model_info
    return models_info


@app.route('/')
def index():
    # return str(app.parsed_data)
    return render_template('index.html')


@click.command()
@click.option('--logs_dir', help='Path to dir where logs are stored', required=True)
@click.option('--host', default='127.0.0.1')
@click.option('--port', default='4200')
def start_app(logs_dir, host, port):
    if not os.path.exists(logs_dir) or not os.path.isdir(logs_dir):
        print(f"No such logs directory: {logs_dir}")
        exit()
    app.logs_dir = logs_dir
    app.parsed_data = parse_logs(logs_dir)
    # TODO: remove debug flag in production
    app.run(host=host, port=port)


if __name__ == '__main__':
    start_app()
