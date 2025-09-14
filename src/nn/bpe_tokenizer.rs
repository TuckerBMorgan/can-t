use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::hash::Hash;
use std::io;
use regex::Regex;

pub struct BPETokenizer {

}

impl BPETokenizer {



    pub fn new(input: String) -> HashMap<Vec<u8>, u8> {
        let vocab_size = 600;
        let regex_pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        let data = fs::read(input).unwrap();
        let data = String::from_utf8(data).unwrap();

        let mut count = 0;

        let mut ranks = HashMap::new();

        for i in 0..(2u8.pow(8)) {
            ranks.insert(vec![i as u8], i);
        }

        // words: Vec<Word>, Word = Vec<Token>, Token = Vec<u8>
        let re = Regex::new(regex_pattern).expect("bad regex");
        let mut words: Vec<Vec<Vec<u8>>> = re
            .find_iter(&data)
            .map(|m| {
                m.as_str()
                    .as_bytes()
                    .iter()
                    .map(|&b| vec![b]) // each initial token is a single byte
                    .collect::<Vec<Vec<u8>>>()
            })
            .collect();



            while ranks.len() < vocab_size {
                // Count adjacent token pairs
                let mut counter: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
                for piece in &words {
                    for pair in piece.windows(2) {
                        // pair: &[Vec<u8>] of length 2
                        let a = pair[0].clone();
                        let b = pair[1].clone();
                        *counter.entry((a, b)).or_insert(0) += 1;
                    }
                }
                if counter.is_empty() {
                    break; // nothing left to merge
                }
        
                // Most frequent pair
                let ((a, b), _) = counter
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .expect("counter empty");
        
                // New merged token bytes = concatenation of the two token byte sequences
                let token_bytes = {
                    let mut t = Vec::with_capacity(a.len() + b.len());
                    t.extend_from_slice(&a);
                    t.extend_from_slice(&b);
                    t
                };
                let token_id = ranks.len();
                ranks.insert(token_bytes.clone(), token_id as u8);
        
                // Merge that pair across all words
                let mut new_words = Vec::with_capacity(words.len());
                for word in &words {
                    let mut new_word: Vec<Vec<u8>> = Vec::with_capacity(word.len());
                    let mut i = 0;
                    while i + 1 < word.len() {
                        if word[i] == a && word[i + 1] == b {
                            new_word.push(token_bytes.clone());
                            i += 2; // IMPORTANT: skip both tokens we just merged
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    if i < word.len() {
                        new_word.push(word[i].clone());
                    }
                    new_words.push(new_word);
                }
                words = new_words;
            }
        
            ranks
    }
}


#[cfg(test)]
mod tests {
    use crate::nn::BPETokenizer;

    #[test]
    pub fn basic() {
     //   let a = BPETokenizer::new("local");
    }
}