pub struct DebuggingOptions {
    pub NaNCheck: bool, // An Option that when set will check if a NaN shows up in the tensor/grad data
                        // this will panic after as well
}

impl DebuggingOptions {
    pub fn default() -> DebuggingOptions {
        DebuggingOptions { NaNCheck: true }
    }
}
