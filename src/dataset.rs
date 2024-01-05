use std::{ fs::File, io::{ BufReader, BufRead} };

use super::network::Vec2D;

pub struct Data {
    pub inputs: Vec2D,
    pub targets: Vec2D
}

impl Data {
    fn parse_csv(name: &str) -> Self {
        fn pad_targets(target: f64) -> Vec<f64> {
            let mut targets = vec![0.0; 10];
            targets[target as usize] = 1.0;
            
            targets
        }

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

        let inputs: Vec2D = lines.iter().map(|line| line[1..].to_vec()).collect();
        let targets: Vec2D = lines.iter().map(|line| pad_targets(line[0])).collect();

        Self {
            inputs,
            targets
        }
    }
}

pub struct Dataset {
    pub train: Data,
    pub test: Data 
}

impl Dataset {
    pub fn parse_csv() -> Self {
        Self {
            train: Data::parse_csv("train"),
            test: Data::parse_csv("test")
        }
    }
}
