#![allow(dead_code)]

use crate::network::utils;
use super::optimizations::{ 
    activations::Activation,
    regularization::{ Regularization, Dropout },
    learning_rate::LearningRate,
    adam::{ AdamHyperParams, Adam },
    early_stopping::EarlyStopping,
    batch::Batch
};

use std::{ fs::File, io::{ BufWriter, Write, BufReader, BufRead} };

pub type Vec2D = Vec<Vec<f64>>;

pub struct HyperParams {
    pub composition: Vec<usize>,
    pub activations: Vec<Activation>,
    pub regularization: Regularization,
    pub learning_rate: LearningRate,
    pub optimizer: AdamHyperParams,
    pub batch_size: usize,
    pub early_stopping: EarlyStopping
}

pub struct Network {
    pub weights: Vec<Vec2D>,
    pub biases: Vec2D,
    pub net_inputs: Vec2D,
    pub outputs: Vec2D,
    pub costs: Vec2D,
    pub optimizer: Adam,
    pub dropout_mask: Vec2D,
    pub batch: Batch,
    pub performance: Vec<f64>,
    pub hyper_params: HyperParams,
}

impl Network {   
    pub fn new(hyper_params: HyperParams) -> Self {
        let composition = &hyper_params.composition;

        assert_eq!(
            composition.len() - 1,
            hyper_params.activations.len(), 
            "ERROR: wrong number of activation functions"
        );

        let zeros_2d_vec = utils::zeros_2d_vec(composition, 1);
        let random_3d_vec = utils::random_3d_vec(&mut rand::thread_rng(), composition);

        Self {
            weights: random_3d_vec,
            biases: zeros_2d_vec.clone(),
            net_inputs: zeros_2d_vec.clone(),
            outputs: zeros_2d_vec.clone(),
            costs: zeros_2d_vec.clone(),
            optimizer: Adam::new(composition),
            dropout_mask: Dropout::init_mask(composition),
            batch: Batch::new(composition),
            performance: Vec::new(),
            hyper_params,
        }
    }

    pub fn save(&self) {        
        let file = File::create("parameters.txt").expect("ERROR: opening file");
        let mut writer = BufWriter::new(file);

        let mut write = |vec: &[f64]| {
            let vec: Vec<String> = vec.iter().map(f64::to_string).collect();

            for num in &vec {
                writeln!(writer, "{num}").expect("ERROR: writing into buffer");
            }
        };

        for biases in &self.biases {
            write(biases);
        }

        for weights in self.weights.iter().flat_map(|weights| weights.iter()) {
            write(weights);
        }

        writer.flush().expect("ERROR: flushing file buffer");
    }

    pub fn load(hyper_params: HyperParams) -> Self {
        let file = File::open("parameters.txt").expect("ERROR: opening file");
        let reader = BufReader::new(file);

        let params: Vec<f64> = reader.lines()
            .map(|line| {
                line.expect("ERROR: reading line")
                    .trim_matches('"')
                    .parse::<f64>().expect("ERROR: parsing f64")
            })
            .collect();

        let mut network = Self::new(hyper_params);
        let mut index = 0;

        for biases in &mut network.biases {
            for bias in &mut *biases {
                *bias = params[index];
                index += 1;
            }
        }

        for weights in &mut network.weights {
            for weights in &mut *weights {
                for weight in &mut *weights {
                    *weight = params[index];
                    index += 1;
                }
            }
        }

        network
    }
}
