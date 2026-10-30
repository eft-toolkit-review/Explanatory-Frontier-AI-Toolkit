import warning_filters  # Configure warnings before importing torch.
"""
Evaluate all models from A1 to A14 and display a results table
"""
import torch
import warnings
from utils import *
from config.config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from module_test import do_test
import pandas as pd


def run_single_model(model_id, model_name, dataset_name, featurePath, model_index, total_models):
    """Evaluate a single model and return its results"""
    # Set a fixed seed
    set_seed(42)

    # load args
    config_file = r"config\config_pretrained.json"
    args = get_config_regression(model_name, dataset_name, config_file)
    args["featurePath"] = featurePath
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # Print model info
    print(f"\n \n [{model_index}/{total_models}] Running {model_id} ({model_name})...")


    # load data and model
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(args['device'])

    # Set to eval mode
    model.eval()

    # Do a dummy forward pass to initialize lazy modules (e.g., BatchNorm in A11_EF_LSTM)
    first_batch = next(iter(dataloader["test"]))
    dummy_text = first_batch['text'][:1].to(args['device'])
    dummy_audio = first_batch['audio'][:1].to(args['device'])
    dummy_vision = first_batch['vision'][:1].to(args['device'])
    with torch.no_grad():
        _ = model(dummy_text, dummy_audio, dummy_vision)

    model_path = rf"pretrained_model/{model_id}_pretrained_model.pth"
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(args['device'])

    # Run evaluation and get results directly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = do_test(args, model, dataloader["test"])

    return result


def main():
    # Define all models
    models = [
        ('A1_GMFN_l', 'A1_Graph_MFN_L'),
        ('A2_GMFN_a', 'A2_Graph_MFN_A'),
        ('A3_GMFN_i', 'A3_Graph_MFN_I'),
        ('A4_GMFN_la', 'A4_Graph_MFN_LA'),
        ('A5_GMFN_li', 'A5_Graph_MFN_LI'),
        ('A6_GMFN_ai', 'A6_Graph_MFN_AI'),
        ('A7_GMFN', 'A7_Graph_MFN'),
        ('A8_GMFN_noM', 'A8_Graph_MFN_noM'),
        ('A9_GMFN_noG', 'A9_Graph_MFN_noG'),
        ('A10_GMFN_noW', 'A10_Graph_MFN_noW'),
        ('A11_EF_LSTM', 'A11_EF_LSTM'),
        ('A12_TFN', 'A12_TFN'),
        ('A13_LMF', 'A13_LMF'),
        ('A14_MFN', 'A14_MFN'),
    ]

    dataset_name = "CEO_trust"
    featurePath = r"data/CEO_trustworthiness_test_features.pkl"

    # Collect all results
    all_results = []
    total_models = len(models)

    for idx, (model_id, model_name) in enumerate(models, 1):
        try:
            result = run_single_model(model_id, model_name, dataset_name, featurePath, idx, total_models)
            # Keep the four required metrics and rename them
            simplified_result = {
                'Model': model_id,
                'Acc': result.get('Acc'),
                'F1_score': result.get('F1_score'),
                'MAE': result.get('MAE'),
                'Corr': result.get('Corr'),
            }
            all_results.append(simplified_result)
            print(f"✓ {model_id} Completed: Acc={simplified_result['Acc']:.4f}, F1_score={simplified_result['F1_score']:.4f}, MAE={simplified_result['MAE']:.4f}, Corr={simplified_result['Corr']:.4f}")
        except Exception as e:
            print(f"✗ {model_id} Failed ({e})")
            all_results.append({
                'Model': model_id,
                'Acc': None,
                'F1_score': None,
                'MAE': None,
                'Corr': None,
            })

    # Create a DataFrame containing the four metrics
    df = pd.DataFrame(all_results)
    columns_order = ['Model', 'Acc', 'F1_score', 'MAE', 'Corr']
    df = df[columns_order]

    # Display the results table
    print("\n" + "=" * 60)
    print("Evaluation results for all models")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save as a CSV file
    output_file = "evaluation_results_all_models.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")

    return df


if __name__ == '__main__':
    main()
