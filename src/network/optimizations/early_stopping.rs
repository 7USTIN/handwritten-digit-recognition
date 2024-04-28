use crate::network::state::Network;

pub struct EarlyStopping {
    pub stability_threshold: f64,
    pub patience: usize
}

impl EarlyStopping {
    pub fn check(network: &mut Network, accuracy: f64) -> bool {
        let EarlyStopping { stability_threshold, patience } = &network.hyper_params.early_stopping;
        
        network.performance.push(accuracy);

        if network.performance.len() >= *patience {
            let recent_performance = &network.performance[(network.performance.len() - patience)..];

            let mut sum_diff = 0.0;

            for index in 1..recent_performance.len() {
                sum_diff += recent_performance[index] - recent_performance[index - 1];
            }

            let mean_diff = sum_diff / *patience as f64;

            return mean_diff <= *stability_threshold
        }

        false
    }
}
