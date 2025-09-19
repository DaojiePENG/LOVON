import os
import random
import numpy as np
import argparse
from datasets import load_dataset, Dataset, DatasetDict

# Read datasets
def read_datasets():
    # Load the base vision-language-motion pair dataset
    data = load_dataset('csv', data_files='templates/01_vison_language_motion_pair_format.csv')['train']
    
    # Load and process synonym dataset
    yolo_class_synonyms = load_dataset('json', data_files='templates/01_yolo_class_synonyms.json')
    yolo_class_synonyms_dict = yolo_class_synonyms['train'][0]
    
    # Load and process threshold dataset
    thresholds = load_dataset('csv', data_files='templates/01_objects_threshold.csv')['train']
    mission_objects_list = thresholds['mission_objects_list']
    wn_thresholds = {obj: thresh for obj, thresh in zip(thresholds['mission_objects_list'], thresholds['wn_threshold'])}
    hn_thresholds = {obj: thresh for obj, thresh in zip(thresholds['mission_objects_list'], thresholds['hn_threshold'])}
    
    return data, mission_objects_list, wn_thresholds, hn_thresholds, yolo_class_synonyms_dict

# Generate random speed values
def generate_speed(mean_speed, std_speed, speed_min, speed_max, decimal_speed):
    v_x = np.random.normal(mean_speed, std_speed)
    v_x = np.clip(v_x, speed_min, speed_max)  # Ensure speed stays within valid range
    return round(v_x, decimal_speed)

# Generate new dataset samples
def generate_new_data(data, mission_objects_list, wn_thresholds, hn_thresholds, yolo_class_synonyms_dict, num_samples, 
                     mean_speed=0.5, std_speed=1.0, speed_min=-8, speed_max=8, decimal_speed=2, 
                     conf_threshold=0.1, k_p=2.0, z_search=0.3):
    mission_states = ['success', 'searching_1', 'searching_0', 'running']
    search_states = ['had_searching_1', 'had_searching_0']
    new_data = []
    
    # Track counts for balanced dataset distribution
    mission_state_counts = {state: 0 for state in mission_states}
    search_state_counts = {state: 0 for state in search_states}
    
    while len(new_data) < num_samples:
        # Randomly select a base row from original data
        row = random.choice(data)
        
        # Generate speed parameter
        v_x = generate_speed(mean_speed, std_speed, speed_min, speed_max, decimal_speed)
        
        # Select mission object and replace with synonym in instruction
        mission_object_1 = random.choice(mission_objects_list)
        if mission_object_1 in yolo_class_synonyms_dict:
            mission_object_1_synonyms = yolo_class_synonyms_dict[mission_object_1]
            random_synonym_1 = random.choice(mission_object_1_synonyms)
        else:
            print(f"Category not found: {mission_object_1}")
            continue  # Skip if category not found
            
        # Replace placeholders in mission instruction
        mission_instruction_1 = row['mission_instruction_1'].replace('mission_object_1', random_synonym_1).replace('v_x', str(v_x))
        
        # Generate mission instruction 0 (2/3 chance to match instruction 1)
        if random.random() < 2/3:
            mission_instruction_0 = mission_instruction_1
            mission_object_0 = mission_object_1  # Keep consistent when instructions match
        else:
            mission_object_0 = random.choice(mission_objects_list)
            v_x_0 = generate_speed(mean_speed, std_speed, speed_min, speed_max, decimal_speed)
            
            if mission_object_0 in yolo_class_synonyms_dict:
                mission_object_0_synonyms = yolo_class_synonyms_dict[mission_object_0]
                random_synonym_0 = random.choice(mission_object_0_synonyms)
            else:
                print(f"Category not found: {mission_object_0}")
                continue  # Skip if category not found
                
            mission_instruction_0 = row['mission_instruction_0'].replace('mission_object_0', random_synonym_0).replace('v_x', str(v_x_0))
        
        # Generate object detection predictions (80% chance to detect target)
        predicted_object = random.choice([mission_object_1, mission_object_1, mission_object_1, mission_object_1, 'NULL'])
        if predicted_object == mission_object_1:
            conf = np.clip(np.random.normal(0.8, 0.2), 0.00, 1.00)  # Confidence around 0.8
            c_xn, c_yn = random.random(), random.random()  # Normalized coordinates
            wn, hn = random.random(), random.random()      # Normalized dimensions
        else:
            conf = 0.0
            c_xn, c_yn = 0.0, 0.0
            wn, hn = 0.0, 0.0
            
        confidence = [conf]
        object_xyn = [c_xn, c_yn]
        object_whn = [wn, hn]
        
        # Determine initial states
        mission_state_in = random.choice(mission_states)
        search_state_in = random.choice(search_states)
        
        # Calculate motion vector based on current state
        if mission_state_in == 'running':
            motion_vector = [v_x, 0.0, -k_p * (c_xn - 0.5)]
        elif mission_state_in == 'searching_1':
            motion_vector = [0.0, 0.0, z_search]
        elif mission_state_in == 'searching_0':
            motion_vector = [0.0, 0.0, -z_search]
        else:  # success state
            motion_vector = [0.0, 0.0, 0.0]
        
        # Determine output states based on conditions
        if mission_instruction_0 != mission_instruction_1:
            mission_state_out = 'searching_1'
            search_state_out = 'had_searching_1'
        elif mission_instruction_0 == mission_instruction_1 and mission_state_in == 'success':
            mission_state_out = 'success'  # Stay successful after completion
            search_state_out = 'had_searching_1'
        elif mission_instruction_0 == mission_instruction_1 and mission_state_in == 'running' and \
             search_state_in == 'had_searching_1' and predicted_object == 'NULL':
            # Lost target from running state - switch to reverse search
            mission_state_out = 'searching_0'
            search_state_out = 'had_searching_0'
        elif mission_instruction_0 == mission_instruction_1 and mission_state_in == 'searching_0' and \
             search_state_in == 'had_searching_0' and predicted_object == 'NULL':
            # Maintain reverse search when target still lost
            mission_state_out = 'searching_0'
            search_state_out = 'had_searching_0'
        elif mission_instruction_0 == mission_instruction_1 and mission_state_in == 'running' and \
             search_state_in == 'had_searching_0' and predicted_object == 'NULL':
            # Lost target from running state - switch to forward search
            mission_state_out = 'searching_1'
            search_state_out = 'had_searching_1'
        elif mission_instruction_0 == mission_instruction_1 and mission_state_in == 'searching_1' and \
             search_state_in == 'had_searching_1' and predicted_object == 'NULL':
            # Maintain forward search when target still lost
            mission_state_out = 'searching_1'
            search_state_out = 'had_searching_1'
        elif mission_instruction_0 == mission_instruction_1 and mission_object_1 == predicted_object and conf >= conf_threshold:
            # Target detected with sufficient confidence
            if wn >= wn_thresholds.get(predicted_object, 0) and hn >= hn_thresholds.get(predicted_object, 0):
                mission_state_out = 'success'
                search_state_out = 'had_searching_1'
            else:
                # Target not yet in optimal position
                if mission_state_in == 'searching_1' and abs(object_xyn[0] - 0.5) > 0.25:
                    mission_state_out = 'searching_1'
                    search_state_out = 'had_searching_1'
                elif mission_state_in == 'searching_0' and abs(object_xyn[0] - 0.5) > 0.25:
                    mission_state_out = 'searching_0'
                    search_state_out = 'had_searching_0'
                else:
                    mission_state_out = 'running'
                    search_state_out = search_state_in  # Maintain previous search state
        else:
            # Maintain current search state for other cases
            if mission_state_in == 'searching_1' and search_state_in == 'had_searching_1':
                mission_state_out = 'searching_1'
                search_state_out = 'had_searching_1'
            elif mission_state_in == 'searching_0' and search_state_in == 'had_searching_0':
                mission_state_out = 'searching_0'
                search_state_out = 'had_searching_0'
            else:
                # Fallback - maintain current state if no conditions matched
                mission_state_out = mission_state_in
                search_state_out = search_state_in
        
        # Balance dataset by ensuring roughly equal distribution of states
        target_count = num_samples / len(mission_states)
        if mission_state_counts[mission_state_out] < target_count and \
           search_state_counts[search_state_in] < (num_samples / len(search_states)):
            
            new_row = {
                'mission_instruction_0': mission_instruction_0,
                'mission_object_0': mission_object_0,
                'mission_instruction_1': mission_instruction_1,
                'mission_object_1': mission_object_1,
                'predicted_object': predicted_object,
                'confidence': confidence,
                'object_xyn': object_xyn,
                'object_whn': object_whn,
                'mission_state_in': mission_state_in,
                'search_state_in': search_state_in,
                'motion_vector': motion_vector,
                'mission_state_out': mission_state_out,
                'search_state_out': search_state_out,
            }
            new_data.append(new_row)
            mission_state_counts[mission_state_out] += 1
            search_state_counts[search_state_in] += 1
        
        # Print progress every 10,000 samples
        if len(new_data) % 10000 == 0 and len(new_data) > 0:
            print(f"Generated {len(new_data)} samples. State distribution: {mission_state_counts}")
    
    return new_data

# Save generated datasets in multiple formats
def save_datasets(new_data, num_samples, r_train=8, r_test=2):
    # Create dataset object
    dataset = Dataset.from_list(new_data)
    
    # Create output directory
    folder_name = f'generated_vlm_dataset_n{num_samples}_cxn025'
    os.makedirs(folder_name, exist_ok=True)
    
    # Save in different formats
    dataset.save_to_disk(os.path.join(folder_name, f'vison_language_motion_pair_format_n{num_samples}'))
    dataset.to_json(os.path.join(folder_name, f'vison_language_motion_pair_format_n{num_samples}.json'))
    
    # Save sample data for inspection
    examples = dataset.select(range(min(100, len(new_data))))  # Handle cases with <100 samples
    examples.to_csv(os.path.join(folder_name, f'vison_language_motion_pair_format_n{num_samples}_examples.csv'))
    
    # Split into train and test sets
    train_test = dataset.train_test_split(train_size=r_train / (r_train + r_test), shuffle=True, seed=42)
    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })
    dataset_dict.save_to_disk(os.path.join(folder_name, f'vison_language_motion_pair_format_split{r_train}{r_test}_n{num_samples}'))
    
    print(f"Dataset saved to {folder_name}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate vision-language-motion paired dataset')
    parser.add_argument('--num_samples', type=int, default=1000000, 
                      help='Number of samples to generate (default: 1,000,000)')
    args = parser.parse_args()
    
    # Execute pipeline
    data, mission_objects_list, wn_thresholds, hn_thresholds, yolo_class_synonyms_dict = read_datasets()
    new_data = generate_new_data(data, mission_objects_list, wn_thresholds, hn_thresholds, 
                                yolo_class_synonyms_dict, args.num_samples)
    save_datasets(new_data, args.num_samples)

if __name__ == "__main__":
    main()
