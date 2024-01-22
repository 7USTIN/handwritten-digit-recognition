use super::{ Network, HyperParams, Vec2D };
use std::{ fs::File, io::{ BufWriter, Write, BufReader, BufRead} };

use rand::{ Rng, rngs::ThreadRng };

#[allow(dead_code)]
impl Network {
    pub fn new(hyper_params: HyperParams) -> Self {
        fn init_weights(rng: &mut ThreadRng, rows: usize, cols: usize) -> Vec2D {
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-1.0..=1.0)).collect())
                .collect::<Vec<Vec<f64>>>()
        }

        let composition = &hyper_params.composition;

        assert_eq!(
            composition.len() - 1,
            hyper_params.activations.len(),
            "ERROR: wrong number of activation functions"
        );
       
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(composition.len() - 1);
        let mut outputs = Vec::with_capacity(composition.len() - 1);

        for index in 1..composition.len() {
            weights.push(init_weights(&mut rng, composition[index], composition[index - 1]));
            outputs.push(vec![0.0; composition[index]]);
        }

        Self {
            weights,
            biases: outputs.clone(),
            net_inputs: outputs.clone(),
            outputs,
            hyper_params
        }
    }

    pub fn save(&self) {        
        let file = File::create("hyper_params.txt").expect("ERROR: opening file");
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
        let file = File::open("hyper_params.txt").expect("ERROR: opening file");
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
