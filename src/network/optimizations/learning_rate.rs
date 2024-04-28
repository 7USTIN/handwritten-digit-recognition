#![allow(dead_code)]

use crate::network::state::Network;

#[derive(Debug)]
pub enum DecayMethod {
    Step(u32),
    Exponential,
    Inverse
}

use DecayMethod::*;

pub struct Decay {
    pub method: DecayMethod,
    pub rate: f64    
}

impl Decay {
    fn decay(decay: &Option<Self>, alpha: &mut f64, adjusted_epoch: u32) {
        if let Some(decay) = decay {
            match decay.method {
                Step(decay_step) => {
                    if adjusted_epoch % decay_step == 0 {
                        *alpha *= decay.rate;
                    }
                },
                Exponential => *alpha *= decay.rate.powi(adjusted_epoch as i32),
                Inverse => *alpha /= 1.0 + decay.rate * adjusted_epoch as f64,
            }            
        }        
    }
}

pub struct Restart {
    pub interval: u32,
    pub alpha: f64,
}

impl Restart {
    fn restart(restart: &Option<Self>, alpha: &mut f64, epoch: &u32) {
        if let Some(restart) = &restart {
            if *epoch % restart.interval == 0 {
                *alpha = restart.alpha;
            }   
        }        
    }
}

pub struct LearningRate {
    pub alpha: f64,
    pub restart: Option<Restart>,
    pub decay: Option<Decay>
}

impl LearningRate {
    pub fn update(network: &mut Network, epoch: &u32) {
        let LearningRate { alpha, decay, restart } = &mut network.hyper_params.learning_rate;

        let mut adjusted_epoch = *epoch;

        if let Some(restart) = &restart {
            adjusted_epoch = epoch % restart.interval;
        }

        Decay::decay(decay, alpha, adjusted_epoch);
        Restart::restart(restart, alpha, epoch);
    }
}
