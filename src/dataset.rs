use super::network::state::Vec2D;

use std::{ fs::File, io::{ BufReader, BufRead} };

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
            .map(|line| line[1..].iter().map(|&value| value / 255.0).collect())
            .collect();
        
        let targets: Vec2D = lines.iter().map(|line| Self::one_hot_encode(line[0] as usize)).collect();

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
