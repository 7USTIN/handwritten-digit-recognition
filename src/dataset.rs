use super::network::state::Vec2D;

use std::{ fs::File, io::{ BufReader, BufRead } };

// 'inputs' consists of a list of handwritten digits
// A digit is represented by 784 grayscale values ranging from 0 (black) - 255 (white)
pub struct Data {
    pub inputs: Vec2D,
    pub targets: Vec2D
}

impl Data { 
    pub fn one_hot_encode(num: usize) -> Vec<f64> {
        let mut vec = vec![0.0; 10];
        vec[num] = 1.0;
        
        vec
    }

    fn parse_csv(name: &str) -> Self {
        let file_name = format!("dataset/mnist_{name}.csv");
        let file = File::open(file_name).expect("ERROR: opening file");
        let reader = BufReader::new(file);

        let lines: Vec2D = reader.lines()
            .map(|line| {
                line.expect("ERROR: reading line")
                    .split(',')
                    .map(|num| num.parse::<f64>().expect("ERROR: parsing f64"))
                    .collect::<Vec<f64>>()
            })
            .collect();

        let inputs: Vec2D = lines
            .iter()
            .map(|line| line[1..].iter().map(|&value| value / 255.0).collect()) // Normalize grayscale values
            .collect();
        
        // One-hot encode targets to allow comparison between target output and actual output.
        let targets: Vec2D = lines.iter().map(|line| Self::one_hot_encode(line[0] as usize)).collect();

        Self {
            inputs,
            targets
        }
    }
}

pub struct Dataset {
    pub train: Data,
    pub validation: Data,
    pub test: Data 
}

impl Dataset {
    pub fn new() -> Self {
        let training_data = Data::parse_csv("train");

        // MNIST Dataset does not provide a validation set, which is why we split the training set 
        let total_lines = training_data.targets.len();
        let validation_start_index = (total_lines as f64 * 0.8) as usize;

        let (train_inputs, validation_inputs) = training_data.inputs.split_at(validation_start_index);
        let (train_targets, validation_targets) = training_data.targets.split_at(validation_start_index);

        Self {
            train: Data { inputs: train_inputs.to_owned(), targets: train_targets.to_owned() },
            validation: Data { inputs: validation_inputs.to_owned(), targets: validation_targets.to_owned() },
            test: Data::parse_csv("test")
        }
    }
}
