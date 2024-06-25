from utils_submission import *
from utils.dataset import process_dataset
import os
import json

class AugmentedSimulator():
    def __init__(self,benchmark):
        self.name = "AirfRANSSubmission"
        #current_dir = '/app/ingested_program/'
        current_dir = os.getcwd()
        self.trainings_dir = os.path.join(current_dir, 'training_baseline')
        #self.trainings_dir = os.path.join(current_dir, trainings_dir_name)
        # Open and read the JSON file
        parameter_path = os.path.join(current_dir,'train_parameters.json')
        #parameter_path = os.path.join(current_dir,train_parameters_file)

        # Create the directory if it does not exist
        os.makedirs(self.trainings_dir, exist_ok=True)
        
        with open(parameter_path, 'r') as file:
            self.cfg = json.load(file)

        with open(os.path.join(self.trainings_dir, 'train_parameters.json'), 'w') as file:
             json.dump(self.cfg, file, indent=4)
        
        self.target_fields = list(self.cfg.keys())
        self.target_fields.remove('regression')

        self.all_outputs = True if 'all_outputs' in self.target_fields else False
        

    def train(self,train_dataset, save_path=None):
        print('Preparing Datasets')
        train_dataset,val_dataset, self.coef_norm = process_dataset(train_dataset, True, coef_norm = None) 
        #val_dataset = process_dataset(test_dataset, False, coef_norm = self.coef_norm)
        print('Starting Global Training')
        if not os.path.exists(self.trainings_dir):
           os.makedirs(self.trainings_dir)
        self.model_dict, self.z_stats, self.global_coef_norm = global_train(self.cfg, self.target_fields, train_dataset, val_dataset,self.trainings_dir)
        print('Finished Global Training')
        

    def predict(self,dataset,**kwargs):

        test_dataset = process_dataset(dataset, False, coef_norm = self.coef_norm) 
        result_test = predict_test(self.cfg, self.model_dict, test_dataset, self.z_stats, self.global_coef_norm,all_outputs=self.all_outputs)

        with open(os.path.join(self.trainings_dir, 'results.txt'), 'a') as file:
            for target_field in self.target_fields:  # Assuming target_fields_output contains the fields you're interested in
                if target_field not in ['implicit_distance','normals']:
                    mse_values = result_test['mse'][target_field]
                    avg_mse = sum(mse_values) / len(mse_values)
                    file.write(f"{target_field} Average MSE: {avg_mse}\n")

        if self.all_outputs:
            # If all_outputs is True, assume result_test['predictions']['all_outputs'] exists and is what we need
            predictions = np.vstack(result_test['predictions']['all_outputs'])*self.coef_norm[3]+ self.coef_norm[2]
            targets = np.vstack(result_test['targets']['all_outputs'])*self.coef_norm[3]+ self.coef_norm[2]
        else:
            # If all_outputs is False, concatenate the predictions for 'p', 'Ux', 'Uy', 'nut'
            concatenated_predictions = []
            concatenated_targets = []
            for sample_idx in range(len(result_test['predictions']['p'])):  # Assuming equal number of samples for each field
                sample_predictions = np.hstack([
                    result_test['predictions']['Ux'][sample_idx],
                    result_test['predictions']['Uy'][sample_idx],
                    result_test['predictions']['p'][sample_idx],
                    result_test['predictions']['nut'][sample_idx]
                ])
                sample_targets = np.hstack([
                    result_test['targets']['Ux'][sample_idx],
                    result_test['targets']['Uy'][sample_idx],
                    result_test['targets']['p'][sample_idx],
                    result_test['targets']['nut'][sample_idx]
                ])
                concatenated_predictions.append(sample_predictions)
                concatenated_targets.append(sample_targets)

            predictions = np.vstack(concatenated_predictions)*self.coef_norm[3]+ self.coef_norm[2]
            targets = np.vstack(concatenated_targets)*self.coef_norm[3]+ self.coef_norm[2]
   
        predictions = dataset.reconstruct_output(predictions)   
        targets= dataset.reconstruct_output(targets)  
       
        return predictions, targets
