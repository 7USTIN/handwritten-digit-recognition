use std::{ fs::File, io::{ BufWriter, Write, BufReader, BufRead} };
use crate::activations::Activation;

use rand::{ Rng, rngs::ThreadRng };

pub type Vec2D = Vec<Vec<f64>>;

pub struct AdamHyperParams {
    pub alpha: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub epsilon: f64
}

pub struct Regularizer {
    pub l1: f64,
    pub l2: f64
}

pub struct Regularization {
    pub weights: Regularizer,
    pub biases: Regularizer
}

pub struct HyperParams {
    pub composition: Vec<usize>,
    pub activations: Vec<Activation>,
    pub regularization: Regularization,
    pub optimizer: AdamHyperParams,
    pub batch_size: usize,
}

#[derive(Clone)]
pub struct Moment {
    pub weights: Vec<Vec2D>,
    pub biases: Vec2D
}

pub struct AdamState {
    pub iteration: i32,
    pub moment_1: Moment,
    pub moment_2: Moment
}

pub struct Batch {
    pub weight_updates: Vec<Vec2D>,
    pub bias_updates: Vec2D
}

pub struct Network {
    pub weights: Vec<Vec2D>,
    pub biases: Vec2D,
    pub net_inputs: Vec2D,
    pub outputs: Vec2D,
    pub costs: Vec2D,
    pub optimizer: AdamState,
    pub batch: Batch,
    pub hyper_params: HyperParams,
}

#[allow(dead_code)]
impl Network {   
    fn random_3d_vec(rng: &mut ThreadRng, composition: &[usize]) -> Vec<Vec2D> {
        let mut random_3d_vec = Vec::with_capacity(composition.len() - 1);
        
        for index in 1..composition.len() {
            random_3d_vec.push(
                (0..composition[index])
                    .map(|_| (0..composition[index - 1]).map(|_| rng.gen_range(-1.0..=1.0)).collect())
                    .collect::<Vec2D>()            
            );
        }

        random_3d_vec
    }

    fn zeros_2d_vec(composition: &[usize]) -> Vec2D {
        let mut zeros_2d_vec = Vec::with_capacity(composition.len() - 1);

        for len in composition.iter().skip(1) {
            zeros_2d_vec.push(vec![0.0; *len]);
        }

        zeros_2d_vec
    }

    fn zeros_3d_vec(composition: &[usize]) -> Vec<Vec2D> {        
        let mut zeros_3d_vec = Vec::with_capacity(composition.len() - 1);
        
        for index in 1..composition.len() {
            let mut layer = Vec::with_capacity(composition[index]);

            for _ in 0..composition[index] {
                layer.push(vec![0.0; composition[index - 1]]);
            }

            zeros_3d_vec.push(layer);
        }

        zeros_3d_vec
    }

    pub fn new(hyper_params: HyperParams) -> Self {
        let composition = &hyper_params.composition;

        assert_eq!(
            composition.len() - 1,
            hyper_params.activations.len(), 
            "ERROR: wrong number of activation functions"
        );

        let zeros_2d_vec = Self::zeros_2d_vec(composition);
        let zeros_3d_vec = Self::zeros_3d_vec(composition);
        let random_3d_vec = Self::random_3d_vec(&mut rand::thread_rng(), composition);

        let moment = Moment {
            weights: zeros_3d_vec.clone(),
            biases: zeros_2d_vec.clone()
        };
        Self {
            weights: random_3d_vec,
            biases: zeros_2d_vec.clone(),
            net_inputs: zeros_2d_vec.clone(),
            outputs: zeros_2d_vec.clone(),
            costs: zeros_2d_vec.clone(),
            optimizer: AdamState {
                iteration: 0,
                moment_1: moment.clone(),
                moment_2: moment,
            },
            batch: Batch {
                weight_updates:  zeros_3d_vec,
                bias_updates: zeros_2d_vec,
            },
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
