pub enum LoggingLevel {
    Standard,
    Advanced,
    Verbose
}

pub struct DebuggingOptions {
    pub nan_check: bool, // An Option that when set will check if a NaN shows up in the tensor/grad data
                         // this will panic after as well
    pub logging_level: LoggingLevel
}

impl DebuggingOptions {
    pub fn default() -> DebuggingOptions {
        DebuggingOptions { 
            nan_check: true,
            logging_level: LoggingLevel::Standard
        }
    }
}
