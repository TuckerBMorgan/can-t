use std::collections::HashMap;
use std::time::Instant;

pub struct Timing {
    lengths: HashMap<String, u128>, // the between with start_clock_with_name and end_clock_with_name was called
    timers: HashMap<String, Instant>, // all current timers we are tracking
}

impl Timing {
    pub fn new() -> Timing {
        Timing {
            lengths: HashMap::new(),
            timers: HashMap::new(),
        }
    }

    /// Starts a clock with a given name
    pub fn start_clock_with_name(&mut self, name: String) {
        self.timers.insert(name, Instant::now());
    }

    /// Ends a clock with a given name, giving it an offical length
    pub fn end_clock_with_name(&mut self, name: String) {
        let start = self.timers[&name].clone();
        let now = Instant::now();
        self.lengths.insert(name, (now - start).as_micros());
    }
}
