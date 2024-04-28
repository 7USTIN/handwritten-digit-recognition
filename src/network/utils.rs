use super::state::Vec2D;
use rand::{ Rng, rngs::ThreadRng };

pub fn random_3d_vec(rng: &mut ThreadRng, composition: &[usize]) -> Vec<Vec2D> {
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

pub fn zeros_2d_vec(composition: &[usize], skip: usize) -> Vec2D {
    let mut zeros_2d_vec = Vec::with_capacity(composition.len() - 1);

    for len in composition.iter().skip(skip) {
        zeros_2d_vec.push(vec![0.0; *len]);
    }

    zeros_2d_vec
}

pub fn zeros_3d_vec(composition: &[usize]) -> Vec<Vec2D> {        
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

